"""Tests for RSA encryption, signing, and key management utilities."""

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from cryptography.exceptions import InvalidSignature
from human_security import HumanRSA

sys.path.insert(1, str(Path(__file__).parent.parent))
from functions.encryption import (
    MAX_CIPHERTEXT_BYTES,
    decode_data,
    decrypt_data,
    deserialize_value,
    encrypt_and_sign_data,
    encrypt_data,
    generate_keys,
    load_private_key,
    load_public_key,
    resolve_key_path,
    save_private_key,
    save_public_key,
    sign_data,
    verify_and_decrypt_data,
    verify_signature,
)


class TestResolveKeyPath:
    """key_path containment guards against directory traversal."""

    def test_contained_path_resolves(self, tmp_path: Path) -> None:
        """A path inside the base resolves to its absolute location."""
        keys = tmp_path / "keys"
        resolved = resolve_key_path(keys, base=tmp_path)
        assert resolved == keys.resolve()

    def test_base_itself_is_allowed(self, tmp_path: Path) -> None:
        """The base directory itself is a valid key_path."""
        assert resolve_key_path(tmp_path, base=tmp_path) == tmp_path.resolve()

    def test_traversal_escape_rejected(self, tmp_path: Path) -> None:
        """A ``../`` path that climbs out of the base is rejected."""
        base = tmp_path / "base"
        base.mkdir()
        with pytest.raises(ValueError, match="escapes the allowed"):
            resolve_key_path(base / ".." / ".." / "etc", base=base)


class TestSecurity:
    """End-to-end security tests: key generation, persistence, crypto ops."""

    def setup_class(self) -> None:
        """Generate sender/receiver key pairs and exchange public keys."""
        self.parent_path = Path(__file__).parent
        self.security_dir = self.parent_path / ".security"
        self.security_dir.mkdir(parents=True, exist_ok=True)
        self.sender, self.receiver = generate_keys()
        sender_pub = self.sender.public_pem()
        receiver_pub = self.receiver.public_pem()
        self.receiver.load_public_pem(sender_pub)
        self.sender.load_public_pem(receiver_pub)

    def teardown_class(self) -> None:
        """Remove PEM key files written during tests."""
        # Delete files if created
        s_pem_pub = self.security_dir / "s_pem.pub"
        if s_pem_pub.exists():
            s_pem_pub.unlink()

        s_pem = self.security_dir / "s_pem"
        if s_pem.exists():
            s_pem.unlink()

        r_pem_pub = self.security_dir / "r_pem.pub"
        if r_pem_pub.exists():
            r_pem_pub.unlink()

        r_pem = self.security_dir / "r_pem"
        if r_pem.exists():
            r_pem.unlink()

    def test_key_generation(self) -> None:
        """Generating sender and receiver key pairs yields non-None objects."""
        assert self.sender is not None
        assert self.receiver is not None

    def test_key_saving_and_loading(self) -> None:
        """Saving and loading public PEM files preserves the key material."""
        save_public_key(self.security_dir / "s_pem.pub", self.sender)
        save_private_key(self.security_dir / "s_pem", self.sender)
        save_public_key(self.security_dir / "r_pem.pub", self.receiver)
        save_private_key(self.security_dir / "r_pem", self.receiver)

        remote_receiver = HumanRSA()
        remote_sender = HumanRSA()
        load_public_key(self.security_dir / "s_pem.pub", remote_receiver)
        load_public_key(self.security_dir / "r_pem.pub", remote_sender)
        assert self.sender.public_pem() == remote_receiver.public_pem()
        assert self.receiver.public_pem() == remote_sender.public_pem()

    def test_private_key_file_is_owner_only(self) -> None:
        """save_private_key writes 0o600 (owner-only) permission bits."""
        priv = self.security_dir / "s_pem"
        save_private_key(priv, self.sender)
        mode = priv.stat().st_mode
        assert mode & 0o777 == 0o600

    def test_private_key_tightens_preexisting_loose_perms(self) -> None:
        """A pre-existing world-readable key file is tightened to 0o600."""
        priv = self.security_dir / "s_pem"
        priv.write_text("stale")
        priv.chmod(0o644)
        save_private_key(priv, self.sender)
        assert priv.stat().st_mode & 0o777 == 0o600

    def test_key_retaining(self) -> None:
        """Saving and loading private PEM files preserves the key material."""
        save_public_key(self.security_dir / "s_pem.pub", self.sender)
        save_private_key(self.security_dir / "s_pem", self.sender)
        save_public_key(self.security_dir / "r_pem.pub", self.receiver)
        save_private_key(self.security_dir / "r_pem", self.receiver)

        remote_receiver = HumanRSA()
        remote_sender = HumanRSA()
        load_private_key(self.security_dir / "s_pem", remote_receiver)
        load_private_key(self.security_dir / "r_pem", remote_sender)
        assert self.sender.private_pem() == remote_receiver.private_pem()
        assert self.receiver.private_pem() == remote_sender.private_pem()

    def test_bytes_encryption_and_decryption(self) -> None:
        """Encrypting then decrypting bytes round-trips to original value."""
        control_action = b"4.20"
        encrypted_c_a = encrypt_data(control_action, self.sender)
        decrypted_c_a = decrypt_data(encrypted_c_a, self.receiver)
        assert control_action == decrypted_c_a

    def test_bytes_signing_and_verification(self) -> None:
        """Signing bytes and verifying against sender's public key succeeds."""
        control_action = b"4.20"
        signature = sign_data(control_action, self.sender)
        verified = verify_signature(control_action, signature, self.receiver)
        assert verified is True

    def test_str_encryption_and_decryption(self) -> None:
        """Encrypting a str round-trips as bytes; decrypt(str) raises error."""
        control_action = "4.20"
        encrypted_c_a = encrypt_data(control_action, self.sender)
        decrypted_c_a = decrypt_data(encrypted_c_a, self.receiver)
        assert control_action.encode("utf-8") == decrypted_c_a
        with pytest.raises(TypeError):
            decrypted_c_a = decrypt_data(control_action, self.receiver)  # type: ignore

    def test_str_signing_and_verification(self) -> None:
        """Signing a str and verifying its UTF-8 bytes against sender's key."""
        control_action = "4.20"
        signature = sign_data(control_action, self.sender)
        verified = verify_signature(
            control_action.encode("utf-8"),
            signature,
            self.receiver,
        )
        assert verified is True

    def test_message_signing_encryption_decryption_and_verification(
        self,
    ) -> None:
        """Signing, encrypting, decrypting, and verifying a dict succeeds."""
        msg = {"a": "1"}
        signed_msg = sign_data(msg, self.sender)
        ciphertext = encrypt_data(signed_msg, self.sender)
        plaintext = decrypt_data(ciphertext, self.receiver)
        # Values travel JSON-serialized, so the decrypted signature must
        # be deserialized before verification.
        sign = deserialize_value(plaintext.pop("signature").decode("utf-8"))
        assert isinstance(sign, str)
        verify = verify_signature(plaintext, sign, self.receiver)
        assert verify is True

    def test_message_signing_encryption_dump_verify_and_decrypt(self) -> None:
        """Encoding to base64, verifying, and decrypting recovers payload."""
        msg = {"a": "1"}
        signed_msg = sign_data(msg, self.sender)
        ciphertext = encrypt_data(signed_msg, self.sender)
        ciphertext_dec = decode_data(ciphertext)
        item = verify_and_decrypt_data(ciphertext_dec, self.receiver)
        assert msg["a"] == item["a"]

    def test_message_signing_encryption_dump_fail_verify(self) -> None:
        """Swapping the signature before decoding raises InvalidSignature."""
        msg = {"a": "1"}
        signed_msg = sign_data(msg, self.sender)
        other_msg = sign_data({"a": "2"}, self.sender)
        signed_msg["signature"] = other_msg["signature"]
        ciphertext = encrypt_data(signed_msg, self.sender)
        ciphertext_str = decode_data(ciphertext)
        with pytest.raises(InvalidSignature):
            verify_and_decrypt_data(ciphertext_str, self.receiver)

    def _round_trip(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Run msg through the full sign-encrypt-dump-verify-decrypt path."""
        signed_msg = sign_data(msg, self.sender)
        ciphertext = encrypt_data(signed_msg, self.sender)
        wire = json.loads(json.dumps(decode_data(ciphertext)))
        return dict(verify_and_decrypt_data(wire, self.receiver))

    def test_round_trip_preserves_float_limits(self) -> None:
        """Univariate result dict round-trips with original value types."""
        msg = {
            "time": "2023-01-01 00:00:00",
            "anomaly": 0,
            "root_cause": None,
            "level_high": 0.5,
            "level_low": -0.5,
        }
        item = self._round_trip(msg)
        assert isinstance(item["level_high"], float)
        assert isinstance(item["level_low"], float)
        assert isinstance(item["anomaly"], int)
        assert isinstance(item["time"], str)
        assert item["root_cause"] is None
        assert item == msg

    def test_round_trip_preserves_dict_limits(self) -> None:
        """Multivariate result dict round-trips limits as dicts."""
        msg = {
            "time": "2023-01-01 00:00:00",
            "anomaly": True,
            "root_cause": "b",
            "level_high": {"a": 0.5, "b": 0.6},
            "level_low": {"a": -0.5, "b": -0.6},
        }
        item = self._round_trip(msg)
        assert isinstance(item["level_high"], dict)
        assert isinstance(item["level_low"], dict)
        assert isinstance(item["anomaly"], bool)
        assert isinstance(item["time"], str)
        assert item == msg

    def test_round_trip_preserves_types_for_chunked_payload(self) -> None:
        """Limits longer than one RSA block are chunked and still parse."""
        msg = {
            "time": "2023-01-01 00:00:00",
            "anomaly": 0,
            "level_high": {f"signal_{i:02d}": i + 0.5 for i in range(30)},
            "level_low": {f"signal_{i:02d}": i - 0.5 for i in range(30)},
        }
        signed_msg = sign_data(msg, self.sender)
        ciphertext = encrypt_data(signed_msg, self.sender)
        # The long limit dicts must exercise the split_msg chunking path.
        assert isinstance(ciphertext["level_high"], list)
        wire = json.loads(json.dumps(decode_data(ciphertext)))
        item = dict(verify_and_decrypt_data(wire, self.receiver))
        assert isinstance(item["level_high"], dict)
        assert isinstance(item["level_low"], dict)
        assert item == msg

    def test_old_format_stringified_payload_still_decrypts(self) -> None:
        """Payloads stringified by the old str() coercion verify and parse."""
        msg = {
            "time": "2023-01-01 00:00:00",
            "anomaly": 0,
            "level_high": 0.5,
        }
        # Replicate the old wire format: every value coerced with str()
        # and encrypted raw, signature computed over the str() dict.
        old_plain = {k: str(v) for k, v in msg.items()}
        signature = self.sender.sign(json.dumps(old_plain).encode("utf-8"))
        old_signed = {**old_plain, "signature": signature}
        ciphertext = {
            k: encrypt_data(v, self.sender) for k, v in old_signed.items()
        }
        wire = json.loads(json.dumps(decode_data(ciphertext)))
        item = dict(verify_and_decrypt_data(wire, self.receiver))
        assert item["time"] == "2023-01-01 00:00:00"
        assert item["anomaly"] == 0
        assert item["level_high"] == 0.5


class TestVerifyFirstFormat:
    """Format version 2: signature verified over ciphertext before decrypt."""

    def setup_class(self) -> None:
        """Generate sender/receiver key pairs and exchange public keys."""
        self.sender, self.receiver = generate_keys()
        sender_pub = self.sender.public_pem()
        receiver_pub = self.receiver.public_pem()
        self.receiver.load_public_pem(sender_pub)
        self.sender.load_public_pem(receiver_pub)

    def test_new_format_round_trip(self) -> None:
        """encrypt_and_sign_data round-trips through verify_and_decrypt."""
        msg = {
            "time": "2023-01-01 00:00:00",
            "anomaly": 0,
            "level_high": 0.5,
            "level_low": -0.5,
        }
        envelope = encrypt_and_sign_data(msg, self.sender)
        assert envelope["format_version"] == "2"
        wire = json.loads(json.dumps(envelope))
        item = dict(verify_and_decrypt_data(wire, self.receiver))
        assert item == msg

    def test_tampered_ciphertext_rejected_before_decrypt(self) -> None:
        """A modified ciphertext fails verification before any RSA decrypt."""
        msg = {"time": "2023-01-01 00:00:00", "anomaly": 0}
        envelope = encrypt_and_sign_data(msg, self.sender)
        wire = json.loads(json.dumps(envelope))
        # Flip a byte in the ciphertext of one field.
        tampered = wire["time"]
        wire["time"] = ("X" if tampered[0] != "X" else "Y") + tampered[1:]
        with (
            patch(
                "functions.encryption.decrypt_data",
            ) as mock_decrypt,
            pytest.raises(InvalidSignature),
        ):
            verify_and_decrypt_data(wire, self.receiver)
        # The private key must never touch attacker-controlled ciphertext.
        mock_decrypt.assert_not_called()

    def test_legacy_accepted_when_allowed_rejected_when_not(self) -> None:
        """A legacy (no format_version) envelope honours the gate flag."""
        msg = {"time": "2023-01-01 00:00:00", "anomaly": 0}
        signed = sign_data(msg, self.sender)
        ciphertext = encrypt_data(signed, self.sender)
        wire = json.loads(json.dumps(decode_data(ciphertext)))
        # Legacy allowed by default.
        item = dict(verify_and_decrypt_data(wire, self.receiver))
        assert item == msg
        # Legacy refused when the downgrade gate is closed.
        with pytest.raises(InvalidSignature, match="downgrade disabled"):
            verify_and_decrypt_data(
                wire,
                self.receiver,
                allow_legacy_signature=False,
            )

    def test_size_cap_rejected_before_rsa(self) -> None:
        """An oversized envelope is rejected before any RSA decryption."""
        oversized: dict[str, str | list[str]] = {
            "payload": "A" * (MAX_CIPHERTEXT_BYTES + 1),
        }
        with (
            patch(
                "functions.encryption.decrypt_data",
            ) as mock_decrypt,
            pytest.raises(ValueError, match="ciphertext"),
        ):
            verify_and_decrypt_data(oversized, self.receiver)
        mock_decrypt.assert_not_called()
