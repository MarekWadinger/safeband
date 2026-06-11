"""RSA encryption, signing, and key management utilities."""

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast, overload

from cryptography.exceptions import InvalidSignature
from human_security import HumanRSA

LEN_LIMIT = 214


def save_public_key(file: str | os.PathLike, key: HumanRSA) -> None:
    """Save the public key to a file.

    Args:
        file (str or Path): Path to the key file.
        key (HumanRSA): Key object containing the public key.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> save_public_key('public_key.pem', key)  # doctest: +SKIP

    """
    with Path(file).open("w") as pub:
        pub.write(key.public_pem())


def save_private_key(file: str | os.PathLike, key: HumanRSA) -> None:
    """Save the private key to a file.

    Args:
        file (str or Path): Path to the key file.
        key (HumanRSA): Key object containing the private key.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> save_private_key('private_key.pem', key)  # doctest: +SKIP

    """
    with Path(file).open("w") as private:
        private.write(key.private_pem())


def load_public_key(file: str | os.PathLike, key: HumanRSA) -> None:
    """Load the public key from a file.

    Args:
        file (str or Path): Path to the public key file.
        key (HumanRSA): Key object to load the public key into.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> load_public_key('public_key.pem', key)  # doctest: +SKIP

    """
    with Path(file).open() as pub:
        key.load_public_pem("".join(pub))


def load_private_key(file: str | os.PathLike, key: HumanRSA) -> None:
    """Load the private key from a file.

    Args:
        file (str or Path): Path to the private key file.
        key (HumanRSA): Key object to load the private key into.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> load_private_key('private_key.pem', key)  # doctest: +SKIP

    """
    with Path(file).open() as pub:
        key.load_private_pem("".join(pub))


@overload
def split_msg(msg: str, max_length: int) -> Sequence[str]: ...
@overload
def split_msg(msg: bytes, max_length: int) -> Sequence[bytes]: ...


def split_msg(msg: str | bytes, max_length: int) -> Sequence[str | bytes]:
    """Split a msg into a list of msgs of specified maximum length.

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
    """Generate a pair of RSA keys.

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
    data: Mapping[str, bytes | str | list[bytes | str]],
    key: HumanRSA,
) -> dict[str, bytes]: ...


def encrypt_data(
    data: str
    | bytes
    | list[bytes]
    | Mapping[str, bytes | str | list[bytes | str]],
    key: HumanRSA,
) -> bytes | list[bytes] | dict[str, bytes]:
    """Encrypt data using the provided key.

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
                v_s = v.decode("utf-8")
            elif not isinstance(v, str):
                v_s = str(v)
            else:
                v_s = v
            data_[x] = encrypt_data(v_s, key)
        return data_
    if isinstance(data, bytes):
        if len(data) > LEN_LIMIT:
            data_ = split_msg(data, LEN_LIMIT)
            return [encrypt_data(d, key) for d in data_]
        return key.encrypt(data)
    return encrypt_data(str(data).encode("utf-8"), key)


@overload
def decrypt_data(data: bytes, key: HumanRSA) -> bytes: ...
@overload
def decrypt_data(data: Sequence[bytes], key: HumanRSA) -> bytes: ...
@overload
def decrypt_data(
    data: Mapping[str, bytes | Sequence[bytes]],
    key: HumanRSA,
) -> dict[str, bytes]: ...


def decrypt_data(
    data: Mapping[str, bytes | Sequence[bytes]] | Sequence[bytes] | bytes,
    key: HumanRSA,
) -> bytes | Sequence[bytes] | dict[str, bytes]:
    """Decrypt data using the provided key.

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
        >>> data = {'a': encrypted_data, 'b': [encrypted_data, encrypted_data]}
        >>> decrypt_data(data, key)
        {'a': b'Test', 'b': b'TestTest'}

    """
    if isinstance(data, dict):
        data_dict = cast("dict[str, bytes | Sequence[bytes]]", data)
        result: dict[str, bytes] = {}
        for x, v in data_dict.items():
            if isinstance(v, bytes):
                result[x] = key.decrypt(v)
            else:
                dec = [key.decrypt(d) for d in v]
                result[x] = b"".join(dec)
        return result
    if isinstance(data, bytes):
        return key.decrypt(data)
    if isinstance(data, list):
        dec = [key.decrypt(d) for d in data]
        return b"".join(dec)
    msg = (
        f"Wrong type of data. Got {type(data)}. Expected (bytes, list, dict)."
    )
    raise TypeError(
        msg,
    )


@overload
def sign_data(data: bytes | str, key: HumanRSA) -> str: ...
@overload
def sign_data(
    data: Mapping[str, bytes | str | int | float],
    key: HumanRSA,
) -> dict[str, bytes | str]: ...


def sign_data(
    data: bytes | str | Mapping[str, bytes | str | int | float],
    key: HumanRSA,
) -> str | dict[str, bytes | str]:
    """Sign the provided data using the given key.

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
    if isinstance(data, Mapping):
        data_map = cast("Mapping[str, bytes | str | int | float]", data)
        data_str: dict[str, str] = {}
        for x, v in data_map.items():
            if isinstance(v, bytes):
                data_str[x] = v.decode("utf-8")
            elif not isinstance(v, str):
                data_str[x] = str(v)
            else:
                data_str[x] = v
        result_dict: dict[str, bytes | str] = {
            k: cast("bytes | str", w) for k, w in data_map.items()
        }
        result_dict["signature"] = key.sign(
            json.dumps(data_str).encode("utf-8"),
        )
        return result_dict
    if isinstance(data, bytes):
        return key.sign(data)
    return key.sign(str(data).encode("utf-8"))


def verify_signature(
    data: str | bytes | Mapping[str, bytes | str],
    signature: str | bytes,
    key: HumanRSA,
) -> bool:
    """Verify the provided signature against the given data and key.

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
    if isinstance(data, Mapping):
        data_map = cast("Mapping[str, bytes | str]", data)
        data_str: dict[str, str] = {
            x: v.decode("utf-8") if isinstance(v, bytes) else v
            for x, v in data_map.items()
        }
        return verify_signature(
            json.dumps(data_str).encode("utf-8"),
            signature,
            key,
        )
    #     return all(
    #         verify_signature(v, signature, key) for v in data.values()
    #     )
    if isinstance(data, str):
        return key.verify(data.encode("utf-8"), signature)
    return key.verify(data, signature)


def verify_and_decrypt_data(
    item: dict[str, str | list[str]],
    key: HumanRSA,
) -> dict[str, str]:
    """Verify the signature of the item, and return the decrypted data.

    Args:
        item: The item to verify and decrypt.
        key: The key object or key used for decryption.

    Raises:
        InvalidSignature: If the signature verification fails.

    Returns:
        dict: The decrypted data (values decoded to str).

    """
    item_enc = cast("dict[str, bytes | Sequence[bytes]]", encode_data(item))
    item_dec: dict[str, bytes] = decrypt_data(item_enc, key)  # type: ignore[assignment]
    sign = item_dec.pop("signature")
    item_str: dict[str, str] = {
        k: v.decode("latin1") if isinstance(v, bytes) else v
        for k, v in item_dec.items()
    }
    verify = verify_signature(item_str, sign, key)
    if verify is not True:
        msg = "Signature verification failed."
        raise InvalidSignature(msg)

    return item_str


def encode_data(
    data: dict[str, str | list[str]],
    encoding: str = "latin1",
) -> dict[str, bytes | list[bytes]]:
    r"""Encode a data by encoding string values to bytes.

    Args:
        data (dict): The data to encode.
        encoding (str): Codec for string values (default ``latin1``).

    Returns:
        dict: The encoded data.

    Examples:
        >>> msg = {
        ...     'key1': 'Hello',
        ...     'key2': ['abcó\x9cÆ', 'xyz']
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
        TypeError: Invalid data in key2

    """
    result: dict[str, bytes | list[bytes]] = {}
    for k, v in data.items():
        if isinstance(v, str):
            result[k] = v.encode(encoding)
        elif isinstance(v, list):
            result[k] = [s.encode(encoding) for s in v]
        else:
            msg = f"Invalid data in {k}"
            raise TypeError(msg)
    return result


type JsonStr = Mapping[str, str | Sequence[str] | JsonStr]
type JsonBytes = Mapping[str, bytes | Sequence[bytes] | JsonBytes]


@overload
def decode_data(
    data: Mapping[str, bytes | list[bytes]],
) -> dict[str, str | list[str]]: ...
@overload
def decode_data(data: JsonBytes) -> JsonStr: ...
@overload
def decode_data(data: Sequence[bytes]) -> Sequence[str]: ...
@overload
def decode_data(data: bytes | complex | str) -> str: ...
def decode_data(
    data: JsonBytes | Sequence | bytes | complex | str,
) -> JsonStr | Sequence[str] | str:
    r"""Decode a data by decoding bytes values to strings.

    Args:
        data (dict): The data to decode.

    Returns:
        dict: The decoded data.

    Examples:
        >>> msg = {
        ...     'key1': b'abc',
        ...     'key2': [b"abc\xf3\x9c\xc6", b"xyz"]
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
        data_dict = cast("JsonBytes", data)
        return cast(
            "JsonStr",
            {k: decode_data(v) for k, v in data_dict.items()},
        )
    if isinstance(data, (list, tuple, range)):
        return cast("Sequence[str]", [decode_data(s) for s in data])
    if isinstance(data, bytes):
        return data.decode("latin1")
    if isinstance(data, (int, float, complex)) or data is None:
        return str(data)
    if isinstance(data, str):
        return data
    msg = (
        f"Wrong type of data. Got {type(data)}. Expected (bytes, list, dict)."
    )
    raise ValueError(
        msg,
    )


def init_rsa_security(key_path: str) -> tuple[HumanRSA, HumanRSA]:
    """Generate or load sender/receiver RSA key pairs at key_path."""
    sender, receiver = generate_keys()
    kp = Path(key_path)
    if not kp.exists():  # pragma: no cover
        kp.mkdir(parents=True, exist_ok=True)
        save_private_key(kp / "sender_pem", sender)
        save_public_key(kp / "sender_pem.pub", sender)
        save_private_key(kp / "receiver_pem", receiver)
        save_public_key(kp / "receiver_pem.pub", receiver)
        load_public_key(kp / "receiver_pem.pub", sender)
    else:  # pragma: no cover
        if (kp / "sender_pem").exists() and (kp / "sender_pem.pub").exists():
            load_private_key(kp / "sender_pem", sender)
            load_public_key(kp / "sender_pem.pub", receiver)
        else:
            save_private_key(kp / "sender_pem", sender)
            save_public_key(kp / "sender_pem.pub", sender)

        if (kp / "receiver_pem").exists() and (
            kp / "receiver_pem.pub"
        ).exists():
            load_private_key(kp / "receiver_pem", receiver)
            load_public_key(kp / "receiver_pem.pub", sender)
        else:
            save_private_key(kp / "receiver_pem", receiver)
            save_public_key(kp / "receiver_pem.pub", receiver)
    return sender, receiver
