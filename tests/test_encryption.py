import os
import pytest
import sys
from pathlib import Path

from human_security import HumanRSA

sys.path.insert(1, str(Path(__file__).parent.parent))
from functions.encryption import (  # noqa: E402
    generate_keys, save_public_key, save_private_key, load_public_key,
    encrypt_data, decrypt_data, sign_data, verify_signature)


class TestSecurity():

    def setup_class(self):
        self.parent_path = Path(__file__).parent
        self.security_dir = self.parent_path / ".security"
        os.makedirs(self.security_dir, exist_ok=True)
        self.controller, self.actuator = generate_keys()
        controller_pub = self.controller.public_pem()
        actuator_pub = self.actuator.public_pem()
        self.actuator.load_public_pem(controller_pub)
        self.controller.load_public_pem(actuator_pub)

    def teardown_class(self):
        # Delete files if created
        if os.path.exists(self.security_dir / "c_pem.pub"):
            os.remove(self.security_dir / "c_pem.pub")

        if os.path.exists(self.security_dir / "c_pem"):
            os.remove(self.security_dir / "c_pem")

        if os.path.exists(self.security_dir / "a_pem.pub"):
            os.remove(self.security_dir / "a_pem.pub")

        if os.path.exists(self.security_dir / "a_pem"):
            os.remove(self.security_dir / "a_pem")

    def test_key_generation(self):
        assert self.controller is not None
        assert self.actuator is not None

    def test_key_saving_and_loading(self):
        save_public_key(self.security_dir / "c_pem.pub", self.controller)
        save_private_key(self.security_dir / "c_pem", self.controller)
        save_public_key(self.security_dir / "a_pem.pub", self.actuator)
        save_private_key(self.security_dir / "a_pem", self.actuator)

        remote_actuator = HumanRSA()
        remote_controller = HumanRSA()
        load_public_key(self.security_dir / "c_pem.pub", remote_actuator)
        load_public_key(self.security_dir / "a_pem.pub", remote_controller)
        assert self.controller.public_pem() == remote_actuator.public_pem()
        assert self.actuator.public_pem() == remote_controller.public_pem()

    def test_bytes_encryption_and_decryption(self):
        control_action = b'4.20'
        encrypted_c_a = encrypt_data(control_action, self.controller)
        decrypted_c_a = decrypt_data(encrypted_c_a, self.actuator)
        assert control_action == decrypted_c_a

    def test_bytes_signing_and_verification(self):
        control_action = b'4.20'
        signature = sign_data(control_action, self.controller)
        verified = verify_signature(
            control_action, signature, self.actuator)
        assert verified is True

    def test_str_encryption_and_decryption(self):
        control_action = '4.20'
        encrypted_c_a = encrypt_data(control_action, self.controller)
        decrypted_c_a = decrypt_data(encrypted_c_a, self.actuator)
        assert control_action.encode('utf-8') == decrypted_c_a
        with pytest.raises(ValueError):
            decrypted_c_a = decrypt_data(control_action, self.actuator)

    def test_str_signing_and_verification(self):
        control_action = '4.20'
        signature = sign_data(control_action, self.controller)
        verified = verify_signature(
            control_action.encode('utf-8'), signature, self.actuator)
        assert verified is True

    def test_message_signing_encryption_decryption_and_verification(self):
        msg = {'a': 1}
        signed_msg = sign_data(msg, self.controller)
        ciphertext = encrypt_data(signed_msg, self.controller)
        plaintext = decrypt_data(ciphertext, self.actuator)
        sign = plaintext.pop('signature')
        verify = verify_signature(plaintext, sign, self.actuator)
        assert verify is True

    def test_message_signing_encryption_decryption_fail(self):
        msg = {'a': 1}
        signed_msg = sign_data(msg, self.controller)
        ciphertext = encrypt_data(signed_msg, self.controller)
        plaintext = decrypt_data(ciphertext, self.actuator)
        sign = plaintext.pop('signature')
        verify = verify_signature(plaintext, sign, self.actuator)
        assert verify is True
