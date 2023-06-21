import json
import os

from pathlib import Path
from human_security import HumanRSA

LEN_LIMIT = 214


def save_public_key(key_path, key):
    """
    Save the public key to a file.

    Args:
        key_path (str or Path): Path to the key file.
        key (HumanRSA): Key object containing the public key.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> save_public_key('public_key.pem', key)  # doctest: +SKIP
    """
    with open(key_path, "w") as pub:
        pub.write(key.public_pem())


def save_private_key(key_path, key):
    """
    Save the private key to a file.

    Args:
        key_path (str or Path): Path to the key file.
        key (HumanRSA): Key object containing the private key.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> save_private_key('private_key.pem', key)  # doctest: +SKIP
    """
    with open(key_path, "w") as private:
        private.write(key.private_pem())


def load_public_key(key_path, key):
    """
    Load the public key from a file.

    Args:
        key_path (str or Path): Path to the public key file.
        key (HumanRSA): Key object to load the public key into.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> load_public_key('public_key.pem', key)  # doctest: +SKIP
    """
    with open(key_path) as pub:
        key.load_public_pem(''.join(pub))


def split_string(string, max_length):
    """
    Split a string into a list of strings of specified maximum length.

    Args:
        string (str): The input string.
        max_length (int): Maximum length of each split string.

    Returns:
        list: List of strings.

    Examples:
        >>> split_string("Hello, World!", 5)
        ['Hello', ', Wor', 'ld!']
    """
    return [string[i:i+max_length] for i in range(0, len(string), max_length)]


def generate_keys():
    """
    Generate a pair of RSA keys.

    Returns:
        tuple: Tuple containing two HumanRSA objects.

    Examples:
        >>> controller, actuator = generate_keys()  # doctest: +SKIP
    """
    controller = HumanRSA()
    controller.generate()
    actuator = HumanRSA()
    actuator.generate()
    return controller, actuator


def encrypt_data(data, key):
    """
    Encrypt data using the provided key.

    Args:
        data (bytes): Data to encrypt.
        key (HumanRSA): Key object to use for encryption.

    Returns:
        bytes: Encrypted data.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> encrypt_data(b'Test', key)  # doctest: +SKIP
    """
    if isinstance(data, dict):
        for x in data:
            data[x] = encrypt_data(data[x], key)
        return data
    elif isinstance(data, bytes):
        if len(data) > LEN_LIMIT:
            data_ = split_string(data, LEN_LIMIT)
            return [encrypt_data(d, controller) for d in data_]
        else:
            return key.encrypt(data)
    else:
        return encrypt_data(str(data).encode('utf-8'), key)


def decrypt_data(data, key):
    """
    Decrypt data using the provided key.

    Args:
        data (bytes): Data to decrypt.
        key (HumanRSA): Key object to use for decryption.

    Returns:
        bytes: Decrypted data.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> encrypted_data = key.encrypt(b'Test')
        >>> decrypt_data(encrypted_data, key)  # doctest: +SKIP
    """
    if isinstance(data, dict):
        for x in data:
            data[x] = decrypt_data(data[x], key)
        return data
    elif isinstance(data, bytes):
        return key.decrypt(data)
    elif isinstance(data, list):
        dec = [decrypt_data(d, actuator) for d in data]
        return b''.join(dec).decode('utf-8')
    else:
        return decrypt_data(str(data).encode('utf-8'), key)


def sign_data(data, key):
    """
    Sign the provided data using the given key.

    Args:
        data (bytes): Data to sign.
        key (HumanRSA): Key object to use for signing.

    Returns:
        bytes: Signature of the data.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> data = b'Test data'
        >>> signature = sign_data(data, key)
        >>> len(signature) > 0
        True
    """
    if isinstance(data, dict):
        for x in data:
            if not isinstance(data[x], str):
                data[x] = str(data[x])
        data["signature"] = key.sign(json.dumps(data).encode('utf-8'))
        return data
    elif isinstance(data, bytes):
        return key.sign(data)
    else:
        return key.sign(str(data).encode('utf-8'))


def verify_signature(data, signature, key):
    """
    Verify the provided signature against the given data and key.

    Args:
        data (bytes): Data to verify.
        signature (bytes): Signature to verify.
        key (HumanRSA): Key object to use for verification.

    Returns:
        bool: True if the signature is valid, False otherwise.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> data = b'Test data'
        >>> signature = key.sign(data)
        >>> verify_signature(data, signature, key)
        True
    """
    if isinstance(data, dict):
        for x in data:
            if isinstance(data[x], bytes):
                data[x] = data[x].decode('utf-8')
        return verify_signature(json.dumps(data).encode('utf-8'),
                                signature, key)
    else:
        return key.verify(data, signature)


# Main code
if __name__ == '__main__':
    control_action = b'4.20'
    parent_path = Path(__file__).parent
    security_dir = parent_path / ".security"
    os.makedirs(security_dir, exist_ok=True)
    # Generate keys for controller and actuator
    controller, actuator = generate_keys()

    # Save keys
    save_public_key(security_dir / "c_pem.pub", controller)
    save_private_key(security_dir / "c_pem", controller)
    save_public_key(security_dir / "a_pem.pub", actuator)
    save_private_key(security_dir / "a_pem", actuator)

    # Load public keys for key exchange
    load_public_key(security_dir / "c_pem.pub", actuator)
    load_public_key(security_dir / "a_pem.pub", controller)

    # Sign control action
    signature = sign_data(control_action, controller)

    # Encrypt control action
    encrypted_c_a = encrypt_data(control_action, controller)

    # Encrypt and split signature if necessary
    if len(signature) > LEN_LIMIT:
        signatures = split_string(signature, LEN_LIMIT)
        encrypted_s_cs = [encrypt_data(bytes(signature_, 'utf-8'), controller)
                          for signature_ in signatures]
    else:
        encrypted_s_cs = encrypt_data(signature, controller)

    # Decrypt control action
    decrypted_c_a = decrypt_data(encrypted_c_a, actuator)

    # Decrypt and join signatures
    decrypted_s_cs = [decrypt_data(encrypted_s_c, actuator)
                      for encrypted_s_c in encrypted_s_cs]
    decrypted_s_c = b''.join(decrypted_s_cs).decode('utf-8')

    # Verify signature
    signature_valid = verify_signature(decrypted_c_a, signature, actuator)

    assert signature_valid

    # Test 2
    data = {'a': 1}
    data_sign = sign_data(data, controller)
    data_sec = encrypt_data(data_sign, controller)
    data_dec = decrypt_data(data_sec, actuator)
    sig = data_dec.pop('signature')
    verify = verify_signature(data_dec, sig, actuator)
    assert verify
