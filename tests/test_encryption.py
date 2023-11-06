import os
import sys
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from human_security import HumanRSA

sys.path.insert(1, str(Path(__file__).parent.parent))
from functions.encryption import (  # noqa: E402
    decode_data,
    decrypt_data,
    encrypt_data,
    generate_keys,
    load_private_key,
    load_public_key,
    save_private_key,
    save_public_key,
    sign_data,
    verify_and_decrypt_data,
    verify_signature,
)


class TestSecurity:
    def setup_class(self):
        self.parent_path = Path(__file__).parent
        self.security_dir = self.parent_path / ".security"
        os.makedirs(self.security_dir, exist_ok=True)
        self.sender, self.receiver = generate_keys()
        sender_pub = self.sender.public_pem()
        receiver_pub = self.receiver.public_pem()
        self.receiver.load_public_pem(sender_pub)
        self.sender.load_public_pem(receiver_pub)

    def teardown_class(self):
        # Delete files if created
        if os.path.exists(self.security_dir / "s_pem.pub"):
            os.remove(self.security_dir / "s_pem.pub")

        if os.path.exists(self.security_dir / "s_pem"):
            os.remove(self.security_dir / "s_pem")

        if os.path.exists(self.security_dir / "r_pem.pub"):
            os.remove(self.security_dir / "r_pem.pub")

        if os.path.exists(self.security_dir / "r_pem"):
            os.remove(self.security_dir / "r_pem")

    def test_key_generation(self):
        assert self.sender is not None
        assert self.receiver is not None

    def test_key_saving_and_loading(self):
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

    def test_key_retaining(self):
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

    def test_bytes_encryption_and_decryption(self):
        control_action = b"4.20"
        encrypted_c_a = encrypt_data(control_action, self.sender)
        decrypted_c_a = decrypt_data(encrypted_c_a, self.receiver)
        assert control_action == decrypted_c_a

    def test_bytes_signing_and_verification(self):
        control_action = b"4.20"
        signature = sign_data(control_action, self.sender)
        verified = verify_signature(control_action, signature, self.receiver)
        assert verified is True

    def test_str_encryption_and_decryption(self):
        control_action = "4.20"
        encrypted_c_a = encrypt_data(control_action, self.sender)
        decrypted_c_a = decrypt_data(encrypted_c_a, self.receiver)
        assert control_action.encode("utf-8") == decrypted_c_a
        with pytest.raises(ValueError):
            decrypted_c_a = decrypt_data(control_action, self.receiver)

    def test_str_signing_and_verification(self):
        control_action = "4.20"
        signature = sign_data(control_action, self.sender)
        verified = verify_signature(
            control_action.encode("utf-8"), signature, self.receiver
        )
        assert verified is True

    def test_message_signing_encryption_decryption_and_verification(self):
        msg = {"a": 1}
        signed_msg = sign_data(msg, self.sender)
        ciphertext = encrypt_data(signed_msg, self.sender)
        plaintext = decrypt_data(ciphertext, self.receiver)
        sign = plaintext.pop("signature")
        verify = verify_signature(plaintext, sign, self.receiver)
        assert verify is True

    def test_message_signing_encryption_dump_verify_and_decrypt(self):
        msg = {"a": 1}
        signed_msg = sign_data(msg, self.sender)
        ciphertext = encrypt_data(signed_msg, self.sender)
        ciphertext = decode_data(ciphertext)
        item = verify_and_decrypt_data(ciphertext, self.receiver)
        assert msg == item

    def test_message_signing_encryption_dump_fail_verify(self):
        msg = {"a": 1}
        signed_msg = sign_data(msg, self.sender)
        other_msg = sign_data({"a": 2}, self.sender)
        signed_msg["signature"] = other_msg["signature"]
        ciphertext = encrypt_data(signed_msg, self.sender)
        ciphertext = decode_data(ciphertext)
        with pytest.raises(InvalidSignature):
            verify_and_decrypt_data(ciphertext, self.receiver)
