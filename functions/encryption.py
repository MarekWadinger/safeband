import json
import os

from cryptography.exceptions import InvalidSignature
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


def load_private_key(key_path, key: HumanRSA):
    """
    Load the private key from a file.

    Args:
        key_path (str or Path): Path to the private key file.
        key (HumanRSA): Key object to load the private key into.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> load_private_key('private_key.pem', key)  # doctest: +SKIP
    """
    with open(key_path) as pub:
        key.load_private_pem(''.join(pub))


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
        >>> sender, receiver = generate_keys()
    """
    sender = HumanRSA()
    sender.generate()
    receiver = HumanRSA()
    receiver.generate()
    return sender, receiver


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


def verify_and_decrypt_data(item, key):
    """
    Verify the signature of the item, and return the decrypted data.

    Args:
        item: The item to verify and decrypt.
        key: The key object or key used for decryption.

    Raises:
        InvalidSignature: If the signature verification fails.

    Returns:
        dict: The decrypted data.
    """
    item = encode_data(item)
    item = decrypt_data(item, key)
    sign = item.pop('signature')
    verify = verify_signature(item, sign, key)
    if verify is not True:
        raise InvalidSignature("Signature verification failed.")

    return item


def encode_data(data):
    """
    Encode a data by encoding string values to bytes.

    Args:
        data (dict): The data to encode.

    Returns:
        dict: The encoded data.

    Examples:
        >>> msg = {
        ...     'key1': 'Hello',
        ...     'key2': ['abcó\\x9cÆ', 'xyz']
        ... }
        >>> encode_data(msg)
        {'key1': b'Hello', 'key2': [b'abc\xf3\x9c\xc6', b'xyz']}

        >>> invalid_msg = {
        ...     'key1': '123',
        ...     'key2': b'Hello'
        ... }
        >>> encode_data(invalid_msg)
        Traceback (most recent call last):
        ...
        ValueError: Invalid data in key2
    """
    for k, v in data.items():
        if isinstance(v, str):
            data[k] = v.encode('latin1')
        elif isinstance(v, list):
            data[k] = [s.encode("latin1") for s in v]
        else:
            raise ValueError(f"Invalid data in {k}")
    return data


def decode_data(data):
    """
    Decode a data by decoding bytes values to strings.

    Args:
        data (dict): The data to decode.

    Returns:
        dict: The decoded data.

    Examples:
        >>> msg = {
        ...     'key1': b'abc',
        ...     'key2': [b"abc\\xf3\\x9c\\xc6", b"xyz"]
        ... }
        >>> decode_data(msg)
        {'key1': 'abc', 'key2': ['abcó\x9cÆ', 'xyz']}

        >>> msg = {
        ...     'key1': 123,
        ...     'key2': b'Hello',
        ...     'key3': 'World',
        ... }
        >>> decode_data(msg)
        {'key1': '123', 'key2': 'Hello', 'key3': 'World'}

        >>> msg = {'key1': type('UnsupportedClass', (), {'value': 42})()}
        >>> decode_data(msg)
        Traceback (most recent call last):
        ...
        ValueError: Wrong type of data. Got <class 'encryption.UnsupportedClass'>. Expected (bytes, list, dict).
    """  # noqa: E501
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = decode_data(v)
        return data
    elif isinstance(data, (list, tuple, range)):
        data = [decode_data(s) for s in data]
        return data
    elif isinstance(data, bytes):
        return data.decode('latin1')
    elif isinstance(data, (int, float, complex)) or data is None:
        return str(data)
    elif isinstance(data, str):
        return data
    else:
        raise ValueError(f"Wrong type of data. Got {type(data)}. "
                         "Expected (bytes, list, dict).")


def init_rsa_security(key_path):
    sender, receiver = generate_keys()
    if not os.path.exists(key_path):  # pragma: no cover
        os.makedirs(key_path, exist_ok=True)
        save_private_key(key_path + "/sender_pem", sender)
        save_public_key(key_path + "/sender_pem.pub", sender)
        save_private_key(key_path + "/receiver_pem", receiver)
        save_public_key(key_path + "/receiver_pem.pub", receiver)
        load_public_key(key_path + "/receiver_pem.pub", sender)
    else:  # pragma: no cover
        if (
                os.path.exists(key_path + "/sender_pem") and
                os.path.exists(key_path + "/sender_pem.pub")
                ):
            load_private_key(key_path + "/sender_pem", sender)
            load_public_key(key_path + "/sender_pem.pub", receiver)
        else:
            save_private_key(key_path + "/sender_pem", sender)
            save_public_key(key_path + "/sender_pem.pub", sender)

        if (
                os.path.exists(key_path + "/receiver_pem") and
                os.path.exists(key_path + "/receiver_pem.pub")
                ):
            load_private_key(key_path + "/receiver_pem", receiver)
            load_public_key(key_path + "/receiver_pem.pub", sender)
        else:
            save_private_key(key_path + "/receiver_pem", receiver)
            save_public_key(key_path + "/receiver_pem.pub", receiver)
    return sender, receiver
