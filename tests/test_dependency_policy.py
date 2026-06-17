"""Enforce the dependency version-constraint policy.

The policy keeps Dependabot churn intentional and CVE patches unimpeded:

* **Never upper-capped** -- security/safety libraries (a fix shipped in a
  new major must install without a manifest edit) and stable, loosely
  coupled cores, typing shims, stubs and plugins.
* **Capped to ``<next-major``** -- libraries whose APIs this project calls
  directly and which break across major releases.

Every runtime dependency must appear in exactly one set below; an
unclassified one fails :func:`test_every_runtime_dependency_is_classified`,
forcing a deliberate decision whenever a dependency is added. The dev,
build and example groups are development-time only and out of scope.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Lower-bound only: security/safety libraries whose patches (even when
# released as a new major) must flow freely, plus stable cores/shims/stubs
# with no expected code-breaking major.
NO_UPPER_CAP = {
    "cryptography",
    "human-security",
    "scipy",
    "typing-extensions",
    "pandas-stubs",
    "types-paho-mqtt",
    "streamz-pulsar",
}
# Capped to <next-major: this project calls APIs that break across the
# major releases of these libraries, so an unreviewed major would break us.
REQUIRE_UPPER_CAP = {
    "pydantic",
    "pandas",
    "paho-mqtt",
    "nats-py",
    "streamz",
    "matplotlib",
    "plotly",
}
# Pinned through [tool.uv.sources] (a VCS fork); the version range in
# [project.dependencies] is not the controlling lever.
VCS_SOURCED = {"river"}


def _canonical(name: str) -> str:
    """Return the PEP 503 normalized form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _runtime_dependencies() -> dict[str, str]:
    """Map each runtime dependency's canonical name to its spec string.

    The spec excludes any environment marker (the part after ``;``).
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in data["project"]["dependencies"]:
        spec = entry.split(";", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9._-]+)(.*)$", spec)
        assert match is not None, f"unparsable dependency: {entry!r}"
        out[_canonical(match.group(1))] = match.group(2).strip()
    return out


def _has_upper_bound(spec: str) -> bool:
    """Whether a marker-stripped requirement carries a ``<`` / ``<=`` cap."""
    return "<" in spec


def test_policy_sets_are_disjoint_and_not_stale() -> None:
    """The three policy sets do not overlap and have no stale entries."""
    assert NO_UPPER_CAP.isdisjoint(REQUIRE_UPPER_CAP)
    assert VCS_SOURCED.isdisjoint(NO_UPPER_CAP | REQUIRE_UPPER_CAP)
    names = set(_runtime_dependencies())
    stale = (NO_UPPER_CAP | REQUIRE_UPPER_CAP | VCS_SOURCED) - names
    assert not stale, f"policy lists dependencies no longer present: {stale}"


def test_every_runtime_dependency_is_classified() -> None:
    """Each runtime dependency is classified in exactly one policy set."""
    names = set(_runtime_dependencies())
    classified = NO_UPPER_CAP | REQUIRE_UPPER_CAP | VCS_SOURCED
    unclassified = names - classified
    assert not unclassified, (
        "Classify these dependencies in tests/test_dependency_policy.py "
        f"(cap only if our code couples to breakable majors): {unclassified}"
    )


def test_security_and_stable_dependencies_are_not_capped() -> None:
    """Never-capped dependencies carry no upper bound, so fixes flow."""
    deps = _runtime_dependencies()
    offenders = {
        name
        for name, spec in deps.items()
        if name in NO_UPPER_CAP and _has_upper_bound(spec)
    }
    assert not offenders, (
        "These must NOT be upper-capped (security/stable; let patches "
        f"flow): {offenders}"
    )


def test_coupled_dependencies_are_capped() -> None:
    """Tightly coupled dependencies are capped to ``<next-major``."""
    deps = _runtime_dependencies()
    offenders = {
        name
        for name, spec in deps.items()
        if name in REQUIRE_UPPER_CAP and not _has_upper_bound(spec)
    }
    assert not offenders, (
        "These couple to APIs that break across majors and must be capped "
        f"<next-major: {offenders}"
    )
