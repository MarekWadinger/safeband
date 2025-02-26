import json
import os
from typing import Mapping, Sequence, overload

from cryptography.exceptions import InvalidSignature
from human_security import HumanRSA

LEN_LIMIT = 214


def save_public_key(file: str | os.PathLike, key: HumanRSA):
    """
    Save the public key to a file.

    Args:
        file (str or Path): Path to the key file.
        key (HumanRSA): Key object containing the public key.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> save_public_key('public_key.pem', key)  # doctest: +SKIP
    """
    with open(file, "w") as pub:
        pub.write(key.public_pem())


def save_private_key(file: str | os.PathLike, key: HumanRSA):
    """
    Save the private key to a file.

    Args:
        file (str or Path): Path to the key file.
        key (HumanRSA): Key object containing the private key.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> save_private_key('private_key.pem', key)  # doctest: +SKIP
    """
    with open(file, "w") as private:
        private.write(key.private_pem())


def load_public_key(file: str | os.PathLike, key: HumanRSA):
    """
    Load the public key from a file.

    Args:
        file (str or Path): Path to the public key file.
        key (HumanRSA): Key object to load the public key into.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> load_public_key('public_key.pem', key)  # doctest: +SKIP
    """
    with open(file) as pub:
        key.load_public_pem("".join(pub))


def load_private_key(file: str | os.PathLike, key: HumanRSA):
    """
    Load the private key from a file.

    Args:
        file (str or Path): Path to the private key file.
        key (HumanRSA): Key object to load the private key into.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> load_private_key('private_key.pem', key)  # doctest: +SKIP
    """
    with open(file) as pub:
        key.load_private_pem("".join(pub))


@overload
def split_msg(msg: str, max_length: int) -> Sequence[str]: ...
@overload
def split_msg(msg: bytes, max_length: int) -> Sequence[bytes]: ...


def split_msg(msg: str | bytes, max_length: int) -> Sequence[str | bytes]:
    """
    Split a msg into a list of msgs of specified maximum length.

    Args:
        msg (str): The input msg.
        max_length (int): Maximum length of each split msg.

    Returns:
        list: List of msgs.

    Examples:
        >>> split_msg("Hello, World!", 5)
        ['Hello', ', Wor', 'ld!']
    """
    return [msg[i : i + max_length] for i in range(0, len(msg), max_length)]


def generate_keys() -> tuple[HumanRSA, HumanRSA]:
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


@overload
def encrypt_data(data: str | bytes, key: HumanRSA) -> bytes: ...
@overload
def encrypt_data(data: list[bytes], key: HumanRSA) -> list[bytes]: ...
@overload
def encrypt_data(
    data: Mapping[str, bytes | str | list[bytes | str]], key: HumanRSA
) -> dict[str, bytes]: ...


def encrypt_data(
    data: str
    | bytes
    | list[bytes]
    | Mapping[str, bytes | str | list[bytes | str]],
    key: HumanRSA,
) -> bytes | list[bytes] | dict[str, bytes]:
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
        >>> print(encrypt_data('Test', key))
        b...
        >>> print(encrypt_data(b'Test', key))
        b...
        >>> print(encrypt_data([b'Test', b'Test'], key))
        b...
        >>> print(encrypt_data({'a': b'Test', 'b': ['Test', b'Test']}, key))
        {'a': b..., 'b': b...}
    """
    if isinstance(data, dict):
        data_ = {}
        for x, v in data.items():
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            elif not isinstance(v, str):
                v = str(v)
            data_[x] = encrypt_data(v, key)
        return data_
    elif isinstance(data, bytes):
        if len(data) > LEN_LIMIT:
            data_ = split_msg(data, LEN_LIMIT)
            return [encrypt_data(d, key) for d in data_]
        else:
            return key.encrypt(data)
    else:
        return encrypt_data(str(data).encode("utf-8"), key)


@overload
def decrypt_data(data: bytes | Sequence[bytes], key: HumanRSA) -> bytes: ...
@overload
def decrypt_data(data: Sequence[bytes], key: HumanRSA) -> Sequence[bytes]: ...
@overload
def decrypt_data(
    data: Mapping[str, bytes | Sequence[bytes]], key: HumanRSA
) -> dict[str, bytes]: ...


def decrypt_data(
    data: Mapping[str, bytes | Sequence[bytes]] | Sequence[bytes] | bytes,
    key: HumanRSA,
) -> bytes | Sequence[bytes] | dict[str, bytes]:
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
        >>> decrypt_data(encrypted_data, key)
        b'Test'
        >>> decrypt_data([encrypted_data, encrypted_data], key)
        b'TestTest'
        >>> decrypt_data({'a': encrypted_data, 'b': [encrypted_data, encrypted_data]}, key)
        {'a': b'Test', 'b': b'TestTest'}
    """
    if isinstance(data, dict):
        # TODO: for some reason dict comprehension doesn't provide correct result
        for x in data:
            data[x] = decrypt_data(data[x], key)
        return data  # type: ignore  # {k: decrypt_data(v, key) for k, v in data.items()}
    elif isinstance(data, bytes):
        return key.decrypt(data)
    elif isinstance(data, list):
        dec = [decrypt_data(d, key) for d in data]
        return b"".join(dec)
    else:
        raise ValueError(
            f"Wrong type of data. Got {type(data)}. Expected (bytes, list, dict)."
        )


@overload
def sign_data(data: bytes | str, key: HumanRSA) -> str: ...
@overload
def sign_data(
    data: Mapping[str, bytes | str], key: HumanRSA
) -> dict[str, bytes | str]: ...


def sign_data(
    data: bytes | str | Mapping[str, bytes | str],
    key: HumanRSA,
) -> str | dict[str, bytes | str]:
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
        >>> signature
        '...'
        >>> sign_data({'a': data, 'b': 'Test data', 'c': 1}, key)
        {'a': b'...', 'b': '...', 'c': 1, 'signature': '...'}
    """
    if isinstance(data, dict):
        data_ = {}
        for x, v in data.items():
            if isinstance(v, bytes):
                data_[x] = v.decode("utf-8")
            elif not isinstance(v, str):
                data_[x] = str(v)
            else:
                data_[x] = v
        data["signature"] = key.sign(json.dumps(data_).encode("utf-8"))
        return data
    elif isinstance(data, bytes):
        return key.sign(data)
    else:
        return key.sign(str(data).encode("utf-8"))


def verify_signature(
    data: str | bytes | Mapping[str, bytes | str],
    signature: str | bytes,
    key: HumanRSA,
) -> bool:
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
        >>> signature
        '...'
        >>> verify_signature(data, signature, key)
        True
        >>> data = {'a': 'Test'}
        >>> signature = key.sign(json.dumps(data).encode("utf-8"))
        >>> verify_signature(data, signature, key)
        True
        >>> verify_signature({'a': data, 'b': b'attack'}, signature, key)
        False
    """
    if isinstance(signature, bytes):
        signature = signature.decode("utf-8")
    if isinstance(data, dict):
        # TODO: for some reason creating new dict doesn't provide correct result
        for x in data:
            if isinstance(data[x], bytes):
                data[x] = data[x].decode("utf-8")
        return verify_signature(
            json.dumps(data).encode("utf-8"), signature, key
        )
    #     return all(verify_signature(v, signature, key) for v in data.values())
    elif isinstance(data, str):
        return key.verify(data.encode("utf-8"), signature)
    else:
        return key.verify(data, signature)


def verify_and_decrypt_data(
    item: Mapping[str, str | list[str]], key: HumanRSA
) -> bytes | dict[str, bytes]:
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
    item_enc = encode_data(item)
    item_dec = decrypt_data(item_enc, key)
    sign = item_dec.pop("signature")
    verify = verify_signature(item_dec, sign, key)
    if verify is not True:
        raise InvalidSignature("Signature verification failed.")

    return item_dec


def encode_data(
    data: Mapping[str, str | list[str]],
    encoding: str = "latin1",
) -> dict[str, bytes | list[bytes]]:
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
        >>> encode_data(msg, encoding='latin1')
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
        # TODO: for some reason creating new dict doesn't provide correct result
        if isinstance(v, str):
            data[k] = v.encode(encoding)
        elif isinstance(v, list):
            data[k] = [s.encode(encoding) for s in v]
        else:
            raise ValueError(f"Invalid data in {k}")
    return data

    @overload
    def decode_data(
        data: Mapping[str, bytes | list[bytes]],
    ) -> dict[str, str | list[str]]: ...
    @overload
    def decode_data(data: Sequence[bytes]) -> Sequence[str]: ...
    @overload
    def decode_data(data: bytes | int | float | complex | str) -> str: ...


JsonStr = Mapping[str, str | Sequence[str] | "JsonStr"]
JsonBytes = Mapping[str, bytes | Sequence[bytes] | "JsonBytes"]


def decode_data(
    data: JsonBytes | Sequence | bytes | int | float | complex | str,
) -> JsonStr | Sequence[str] | str:
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
        return {k: decode_data(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple, range)):
        data = [decode_data(s) for s in data]
        return data
    elif isinstance(data, bytes):
        return data.decode("latin1")
    elif isinstance(data, (int, float, complex)) or data is None:
        return str(data)
    elif isinstance(data, str):
        return data
    else:
        raise ValueError(
            f"Wrong type of data. Got {type(data)}. "
            "Expected (bytes, list, dict)."
        )


def init_rsa_security(key_path: str) -> tuple[HumanRSA, HumanRSA]:
    sender, receiver = generate_keys()
    if not os.path.exists(key_path):  # pragma: no cover
        os.makedirs(key_path, exist_ok=True)
        save_private_key(key_path + "/sender_pem", sender)
        save_public_key(key_path + "/sender_pem.pub", sender)
        save_private_key(key_path + "/receiver_pem", receiver)
        save_public_key(key_path + "/receiver_pem.pub", receiver)
        load_public_key(key_path + "/receiver_pem.pub", sender)
    else:  # pragma: no cover
        if os.path.exists(key_path + "/sender_pem") and os.path.exists(
            key_path + "/sender_pem.pub"
        ):
            load_private_key(key_path + "/sender_pem", sender)
            load_public_key(key_path + "/sender_pem.pub", receiver)
        else:
            save_private_key(key_path + "/sender_pem", sender)
            save_public_key(key_path + "/sender_pem.pub", sender)

        if os.path.exists(key_path + "/receiver_pem") and os.path.exists(
            key_path + "/receiver_pem.pub"
        ):
            load_private_key(key_path + "/receiver_pem", receiver)
            load_public_key(key_path + "/receiver_pem.pub", sender)
        else:
            save_private_key(key_path + "/receiver_pem", receiver)
            save_public_key(key_path + "/receiver_pem.pub", receiver)
    return sender, receiver
