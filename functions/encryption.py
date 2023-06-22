import json

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
        >>> controller, actuator = generate_keys()
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
        >>> ciphertext = encrypt_data(b'Test', key)
    """
    if isinstance(data, dict):
        for x in data:
            data[x] = encrypt_data(data[x], key)
        return data
    elif isinstance(data, bytes):
        if len(data) > LEN_LIMIT:
            data_ = split_string(data, LEN_LIMIT)
            return [encrypt_data(d, key) for d in data_]
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
        >>> plaintext = decrypt_data(encrypted_data, key)
    """
    if isinstance(data, dict):
        for x in data:
            data[x] = decrypt_data(data[x], key)
        return data
    elif isinstance(data, bytes):
        return key.decrypt(data)
    elif isinstance(data, list):
        dec = [decrypt_data(d, key) for d in data]
        return b''.join(dec).decode('utf-8')
    else:
        raise ValueError(f"Wrong type of data. Got {type(data)}. "
                         "Expected (bytes, list, dict).")


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
