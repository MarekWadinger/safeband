"""Tests for hardened recovery model save/load (perms + HMAC pickle)."""

import datetime as dt
import sys
from pathlib import Path

from river import proba, utils

sys.path.insert(1, str(Path(__file__).parent.parent))

from functions.anomaly import GaussianScorer
from functions.model_persistence import load_model, save_model


def _make_model() -> GaussianScorer:
    """Build a minimal GaussianScorer suitable for round-trip tests."""
    day = dt.timedelta(days=1)
    return GaussianScorer(
        utils.TimeRolling(proba.Gaussian(), period=day),
        grace_period=day,
    )


class TestRecoveryRoundTrip:
    """A model written by save_model is loadable via load_model."""

    def test_save_then_load_round_trips(self, tmp_path: Path) -> None:
        """save_model + load_model recovers the same topics' model."""
        recovery = tmp_path / "recovery"
        save_model(str(recovery), ["plant/a"], _make_model())
        loaded = load_model(str(recovery), ["plant/a"])
        assert loaded is not None

    def test_save_creates_owner_only_dir(self, tmp_path: Path) -> None:
        """save_model creates the recovery dir with 0o700 permissions."""
        recovery = tmp_path / "recovery"
        save_model(str(recovery), ["plant/a"], _make_model())
        assert recovery.stat().st_mode & 0o777 == 0o700


class TestRecoverySecurity:
    """Pickle-RCE hardening: perms refusal and HMAC tamper rejection."""

    def test_world_writable_dir_refused(self, tmp_path: Path) -> None:
        """A world-writable recovery dir is refused at load time."""
        recovery = tmp_path / "recovery"
        save_model(str(recovery), ["plant/a"], _make_model())
        recovery.chmod(0o707)
        assert load_model(str(recovery), ["plant/a"]) is None

    def test_tampered_pickle_rejected(self, tmp_path: Path) -> None:
        """A pickle whose bytes were modified fails HMAC and is skipped."""
        recovery = tmp_path / "recovery"
        save_model(str(recovery), ["plant/a"], _make_model())
        pkl = next(recovery.glob("model_*.pkl"))
        data = bytearray(pkl.read_bytes())
        data[-1] ^= 0xFF
        pkl.write_bytes(bytes(data))
        assert load_model(str(recovery), ["plant/a"]) is None

    def test_missing_mac_rejected(self, tmp_path: Path) -> None:
        """A foreign pickle without a valid HMAC sidecar is not loaded."""
        recovery = tmp_path / "recovery"
        save_model(str(recovery), ["plant/a"], _make_model())
        mac = next(recovery.glob("model_*.pkl.hmac"))
        mac.unlink()
        assert load_model(str(recovery), ["plant/a"]) is None

    def test_foreign_key_rejected(self, tmp_path: Path) -> None:
        """A pickle MAC'd under a different key is rejected at load."""
        recovery = tmp_path / "recovery"
        save_model(str(recovery), ["plant/a"], _make_model())
        # Overwrite the per-dir key so the stored MAC no longer matches.
        (recovery / ".recovery_hmac_key").write_bytes(b"\x00" * 32)
        assert load_model(str(recovery), ["plant/a"]) is None
