"""Joblib-based model save/load helpers for versioned recovery files.

Recovery objects are river-based scorers (:class:`GaussianScorer` /
``ConditionalGaussianScorer``) whose state is a deep graph of rolling
windows, deques and distribution estimators. A clean JSON round-trip with
explicit state reconstruction is impractical and fragile for that graph,
so the pickle format is kept but hardened against the pickle-RCE vector in
two layers:

1. The recovery directory is created ``0o700`` and refused at load time if
   it is world-writable or not owned by the current user, so an attacker
   cannot drop a malicious pickle for us to load.
2. Every pickle is accompanied by an HMAC (keyed with a per-directory
   secret stored ``0o600`` inside the recovery dir). ``load_model``
   verifies the MAC before unpickling, so only files this process wrote
   (with the matching key) are ever deserialized; a tampered or foreign
   pickle is rejected before ``joblib.load`` runs.
"""

import datetime as dt
import hmac
import logging
import os
import secrets
from hashlib import sha256
from pathlib import Path

import joblib

from safeband.anomaly import GaussianScorer
from safeband.utils import common_prefix

logger = logging.getLogger(__name__)

_MAC_SUFFIX = ".hmac"
_KEY_FILENAME = ".recovery_hmac_key"


def _load_or_create_mac_key(recovery_dir: Path) -> bytes:
    """Return the per-directory HMAC key, creating it 0o600 if absent.

    Args:
        recovery_dir: The recovery directory holding the key file.

    Returns:
        bytes: The secret HMAC key.

    """
    key_path = recovery_dir / _KEY_FILENAME
    if key_path.exists():
        return key_path.read_bytes()
    key = secrets.token_bytes(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(key_path, flags, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


def _is_safe_recovery_dir(recovery_dir: Path) -> bool:
    """Report whether ``recovery_dir`` is safe to load pickles from.

    The directory must be owned by the current user and must not be
    world- or group-writable, otherwise another user could plant a
    malicious recovery file.

    Args:
        recovery_dir: The directory to inspect.

    Returns:
        bool: True if the directory is owner-only writable and owned by us.

    """
    st = recovery_dir.stat()
    if st.st_uid != os.geteuid():
        logger.warning(
            "Refusing recovery dir %s: not owned by current user.",
            recovery_dir,
        )
        return False
    if st.st_mode & 0o022:
        logger.warning(
            "Refusing recovery dir %s: group/world-writable.",
            recovery_dir,
        )
        return False
    return True


def load_model(path: str, topics: list[str]) -> GaussianScorer | None:
    """Load a model from a given path.

    Only HMAC-verified pickles from a directory owned by the current user
    and not group/world-writable are deserialized; any file failing those
    checks is skipped to avoid the pickle-RCE vector.

    Args:
        path: The path to the model.
        topics: The topics of the model.

    """
    if not path:
        return None
    recovery_dir = Path(path)
    if not recovery_dir.exists():
        logger.info("No model files found in the recovery folder.")
        return None
    if not _is_safe_recovery_dir(recovery_dir):
        return None
    model_name = f"model_{common_prefix(topics).replace('/', '_')}_*.pkl"
    model_files = sorted(recovery_dir.glob(model_name), reverse=True)
    if not model_files:
        logger.info("No model files found in the recovery folder.")
        return None
    key = _load_or_create_mac_key(recovery_dir)
    for latest_model in model_files:
        if not _verify_mac(latest_model, key):
            logger.warning(
                "Skipping %s: HMAC verification failed.",
                latest_model,
            )
            continue
        recovery_data = joblib.load(latest_model)
        if recovery_data["topics"] == topics:
            logger.info("Latest model found: %s", latest_model)
            return recovery_data["model"]
    logger.info("No matching model files found in the recovery folder.")
    return None


def _verify_mac(pickle_path: Path, key: bytes) -> bool:
    """Verify the HMAC sidecar for ``pickle_path`` against ``key``.

    Args:
        pickle_path: Path to the pickle whose MAC is checked.
        key: The secret HMAC key.

    Returns:
        bool: True if a sidecar exists and matches the pickle's digest.

    """
    mac_path = pickle_path.with_name(pickle_path.name + _MAC_SUFFIX)
    if not mac_path.exists():
        return False
    expected = hmac.new(key, pickle_path.read_bytes(), sha256).digest()
    return hmac.compare_digest(expected, mac_path.read_bytes())


def save_model(
    path: str,
    topics: list[str],
    model: object,
    keep_last: int = 5,
) -> None:
    """Save a model to a given path.

    The recovery directory is created ``0o700`` (owner-only) and each
    pickle is written with an HMAC sidecar so that :func:`load_model` only
    deserializes files this process produced.

    Args:
        path: The path to the model.
        topics: The topics of the model.
        model: The model to save.
        keep_last: Number of newest recovery pickles to retain for this
            topic prefix; older ones are deleted after a successful
            save. Non-positive values disable pruning.

    """
    if not path:
        return
    model_prefix = f"model_{common_prefix(topics).replace('/', '_')}"
    now = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, mode=0o700)
    else:
        p.chmod(0o700)
    key = _load_or_create_mac_key(p)
    recovery_path = p / f"{model_prefix}_{now}.pkl"
    with recovery_path.open("wb") as f:
        joblib.dump({"model": model, "topics": topics}, f)
        logger.info("Model saved to %s", recovery_path)
    mac = hmac.new(key, recovery_path.read_bytes(), sha256).digest()
    mac_path = recovery_path.with_name(recovery_path.name + _MAC_SUFFIX)
    mac_path.write_bytes(mac)
    if keep_last > 0:
        # Timestamped names sort lexicographically, newest first.
        stale = sorted(p.glob(f"{model_prefix}_*.pkl"), reverse=True)[
            keep_last:
        ]
        for old in stale:
            old.unlink()
            old_mac = old.with_name(old.name + _MAC_SUFFIX)
            if old_mac.exists():
                old_mac.unlink()
            logger.info("Pruned old recovery file %s", old)
