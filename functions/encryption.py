"""RSA encryption, signing, and key management utilities."""

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast, overload

from cryptography.exceptions import InvalidSignature
from human_security import HumanRSA

LEN_LIMIT = 214

# Envelope wire-format marker. Version 2 signs the CIPHERTEXT so the
# signature is verified before any RSA decryption (decrypt-after-verify).
# Absence of this field means the legacy format that signs the plaintext
# (verify-after-decrypt); see :func:`verify_and_decrypt_data`.
FORMAT_VERSION = 2
FORMAT_VERSION_FIELD = "format_version"

# Defense-in-depth caps applied before any RSA work on a received
# envelope, to bound work an attacker can force per message.
MAX_FIELD_COUNT = 256
MAX_CIPHERTEXT_BYTES = 1 << 20  # 1 MiB of latin1 ciphertext per envelope


def serialize_value(value: object) -> str:
    """Serialize one payload value to its JSON wire representation.

    Bytes pass through as UTF-8 text; every other value is JSON-encoded so
    floats, dicts, ints, bools, and ``None`` keep their types across the
    sign -> encrypt -> decrypt -> verify round trip. Objects JSON cannot
    represent (e.g. ``datetime``) fall back to ``str()``.

    Args:
        value: Payload value to serialize.

    Returns:
        str: Wire representation of the value.

    Examples:
        >>> serialize_value(0.5)
        '0.5'
        >>> serialize_value({"a": 0.5})
        '{"a": 0.5}'
        >>> serialize_value("text")
        '"text"'
        >>> serialize_value(b"raw")
        'raw'

    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return json.dumps(value, default=str)


def deserialize_value(value: str) -> object:
    """Parse a wire value back to its original type.

    Values that are not valid JSON — timestamps such as
    ``"2023-01-01 00:00:00"``, or payloads produced by older versions of
    this module that coerced values with ``str()`` — are returned
    unchanged as strings.

    Args:
        value: Wire representation of a payload value.

    Returns:
        object: The parsed value, or the input string if not valid JSON.

    Examples:
        >>> deserialize_value('0.5')
        0.5
        >>> deserialize_value('{"a": 0.5}')
        {'a': 0.5}
        >>> deserialize_value('2023-01-01 00:00:00')
        '2023-01-01 00:00:00'

    """
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


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
    """Save the private key to a file with owner-only permissions.

    The file is created with mode ``0o600`` (owner read/write only) via
    :func:`os.open` so the private key material is never briefly readable
    by other users at the default umask before a post-hoc ``chmod``.

    Args:
        file (str or Path): Path to the key file.
        key (HumanRSA): Key object containing the private key.

    Examples:
        >>> from human_security import HumanRSA
        >>> key = HumanRSA()
        >>> key.generate()
        >>> save_private_key('private_key.pem', key)  # doctest: +SKIP

    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(file, flags, 0o600)
    # Enforce 0o600 on the open descriptor so a pre-existing file with
    # looser permissions (O_CREAT does not reset those) is tightened too.
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as private:
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
    data: Mapping[str, object],
    key: HumanRSA,
) -> dict[str, bytes]: ...


def encrypt_data(
    data: str | bytes | list[bytes] | Mapping[str, object],
    key: HumanRSA,
) -> bytes | list[bytes] | dict[str, bytes]:
    """Encrypt data using the provided key.

    Dict values are serialized with :func:`serialize_value` (JSON; bytes
    pass through as UTF-8 text, ``datetime`` objects fall back to
    ``str()``) so the consumer can restore their original types after
    decryption. This must stay consistent with :func:`sign_data`, which
    signs the same wire representation.

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
        >>> print(encrypt_data({'a': b'Test', 'b': ['Test', 'Test']}, key))
        {'a': b..., 'b': b...}

    """
    if isinstance(data, dict):
        data_ = {}
        for x, v in data.items():
            data_[x] = encrypt_data(serialize_value(v), key)
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
    data: Mapping[str, object],
    key: HumanRSA,
) -> dict[str, object]: ...


def sign_data(
    data: bytes | str | Mapping[str, object],
    key: HumanRSA,
) -> str | dict[str, object]:
    """Sign the provided data using the given key.

    Dict values are signed in their :func:`serialize_value` wire form
    (JSON; ``datetime`` objects fall back to ``str()``), the same form
    :func:`encrypt_data` encrypts, so the consumer can verify the
    signature against the decrypted wire strings.

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
        data_map = cast("Mapping[str, object]", data)
        data_ser: dict[str, str] = {
            x: serialize_value(v) for x, v in data_map.items()
        }
        result_dict: dict[str, object] = dict(data_map)
        result_dict["signature"] = key.sign(
            json.dumps(data_ser).encode("utf-8"),
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
    if isinstance(data, str):
        return key.verify(data.encode("utf-8"), signature)
    return key.verify(data, signature)


def encrypt_and_sign_data(
    data: Mapping[str, object],
    key: HumanRSA,
) -> dict[str, str | list[str]]:
    """Encrypt ``data`` then sign the ciphertext (format version 2).

    This is the producer counterpart of the decrypt-after-verify path:
    the payload is encrypted first, the latin1-decoded ciphertext wire
    strings are signed, and a :data:`FORMAT_VERSION` marker is added.
    The consumer can therefore verify the signature over the ciphertext
    before performing any RSA decryption.

    Args:
        data: The plaintext payload mapping to protect.
        key: The signing/encrypting key (the sender's key).

    Returns:
        dict: A wire envelope of latin1 ciphertext strings plus a
        ``signature`` field and a ``format_version`` marker.

    """
    ciphertext = encrypt_data(data, key)
    wire = decode_data(ciphertext)
    # Sign the ciphertext wire strings (excluding the marker we are about
    # to add) so the consumer verifies before decrypting.
    signature = key.sign(json.dumps(wire).encode("utf-8"))
    envelope: dict[str, str | list[str]] = dict(wire)
    envelope["signature"] = (
        signature if isinstance(signature, str) else signature.decode("utf-8")
    )
    envelope[FORMAT_VERSION_FIELD] = str(FORMAT_VERSION)
    return envelope


def _check_envelope_caps(item: Mapping[str, object]) -> None:
    """Reject oversized envelopes before any RSA work (defense-in-depth).

    Args:
        item: The received wire envelope.

    Raises:
        ValueError: If the field count or total ciphertext size exceeds
            the configured caps.

    """
    if len(item) > MAX_FIELD_COUNT:
        msg = f"Envelope has {len(item)} fields (cap {MAX_FIELD_COUNT})."
        raise ValueError(msg)
    total = 0
    for v in item.values():
        if isinstance(v, str):
            total += len(v)
        elif isinstance(v, list):
            total += sum(len(s) for s in v if isinstance(s, str))
    if total > MAX_CIPHERTEXT_BYTES:
        msg = (
            f"Envelope ciphertext {total} bytes (cap {MAX_CIPHERTEXT_BYTES})."
        )
        raise ValueError(msg)


def verify_and_decrypt_data(
    item: dict[str, str | list[str]],
    key: HumanRSA,
    *,
    allow_legacy_signature: bool = True,
) -> dict[str, object]:
    """Verify an envelope's signature, then return the decrypted data.

    For format-version-2 envelopes the signature is verified over the
    *ciphertext* BEFORE any RSA decryption, so attacker-controlled
    ciphertext is never fed to the private key unless it is authentic.
    Envelopes without a ``format_version`` marker use the legacy path
    that decrypts first and verifies the plaintext; that path is gated by
    ``allow_legacy_signature`` so a downgrade can be refused.

    Each decrypted value is parsed with :func:`deserialize_value`, so
    floats, dicts, ints, bools, and ``None`` come back with the types the
    producer sent. Values that are not valid JSON — timestamps and
    payloads produced by older versions that coerced values with
    ``str()`` — are returned as strings.

    Rollout note: producers and consumers must co-upgrade. While legacy
    producers remain deployed keep ``allow_legacy_signature=True`` (the
    default); once every producer emits format version 2, set it to
    ``False`` to refuse downgrade to the verify-after-decrypt path.

    Args:
        item: The wire envelope to verify and decrypt.
        key: The key object used for verification and decryption.
        allow_legacy_signature: When True (default) accept legacy
            verify-after-decrypt envelopes; when False reject them.

    Raises:
        InvalidSignature: If signature verification fails, or a legacy
            envelope arrives while ``allow_legacy_signature`` is False.
        ValueError: If the envelope exceeds the size/field-count caps.

    Returns:
        dict: The decrypted data with original value types restored.

    """
    _check_envelope_caps(item)
    if str(item.get(FORMAT_VERSION_FIELD, "")) == str(FORMAT_VERSION):
        return _verify_then_decrypt(item, key)
    if not allow_legacy_signature:
        msg = "Legacy-format signature rejected (downgrade disabled)."
        raise InvalidSignature(msg)
    return _decrypt_then_verify(item, key)


def _verify_then_decrypt(
    item: Mapping[str, str | list[str]],
    key: HumanRSA,
) -> dict[str, object]:
    """Verify the ciphertext signature first, then decrypt (version 2)."""
    wire: dict[str, str | list[str]] = {
        k: v
        for k, v in item.items()
        if k not in ("signature", FORMAT_VERSION_FIELD)
    }
    sign = item["signature"]
    if not isinstance(sign, str):
        msg = "Malformed signature field."
        raise InvalidSignature(msg)
    if key.verify(json.dumps(wire).encode("utf-8"), sign) is not True:
        msg = "Signature verification failed."
        raise InvalidSignature(msg)
    item_enc = cast("dict[str, bytes | Sequence[bytes]]", encode_data(wire))
    item_dec: dict[str, bytes] = decrypt_data(item_enc, key)  # type: ignore[assignment]
    return {
        k: deserialize_value(
            v.decode("latin1") if isinstance(v, bytes) else v,
        )
        for k, v in item_dec.items()
    }


def _decrypt_then_verify(
    item: dict[str, str | list[str]],
    key: HumanRSA,
) -> dict[str, object]:
    """Legacy path: decrypt first, then verify the plaintext signature."""
    item_enc = cast("dict[str, bytes | Sequence[bytes]]", encode_data(item))
    item_dec: dict[str, bytes] = decrypt_data(item_enc, key)  # type: ignore[assignment]
    item_ser: dict[str, str] = {
        k: v.decode("latin1") if isinstance(v, bytes) else v
        for k, v in item_dec.items()
    }
    sign_ser = item_ser.pop("signature")
    # New-format signatures arrive JSON-quoted; old-format ones are raw.
    sign = deserialize_value(sign_ser)
    if not isinstance(sign, str):
        sign = sign_ser
    verify = verify_signature(item_ser, sign, key)
    if verify is not True:
        msg = "Signature verification failed."
        raise InvalidSignature(msg)

    return {k: deserialize_value(v) for k, v in item_ser.items()}


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


def resolve_key_path(
    key_path: str | os.PathLike,
    base: str | os.PathLike | None = None,
) -> Path:
    """Resolve ``key_path`` and reject directory-traversal escapes.

    The path is resolved to an absolute, symlink-free location. When
    ``base`` is given the resolved path must stay inside it, so a value
    such as ``../../etc`` that climbs out of the allowed directory is
    rejected before any key material is read or written.

    Args:
        key_path: User- or config-supplied path to the key directory.
        base: Directory the resolved path must remain within. Defaults to
            the current working directory.

    Returns:
        Path: The safe, resolved absolute path.

    Raises:
        ValueError: If the resolved path escapes ``base``.

    """
    base_resolved = Path(base).resolve() if base is not None else Path.cwd()
    resolved = Path(key_path).resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        msg = f"key_path {key_path!r} escapes the allowed directory."
        raise ValueError(msg)
    return resolved


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
