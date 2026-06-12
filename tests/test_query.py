"""Tests for MQTT message consumption and model persistence utilities."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt
import pytest
from human_security import HumanRSA

sys.path.insert(1, str(Path(__file__).parent.parent))
from typing import TYPE_CHECKING

from consumer import on_message, query_file
from functions.encryption import (
    decode_data,
    encrypt_data,
    sign_data,
)
from functions.model_persistence import load_model, save_model
from functions.utils import common_prefix

if TYPE_CHECKING:
    from functions.typing_extras import FileClient


class TestConsumer:
    """Tests for the MQTT on_message handler and file-based query path."""

    def setup_class(self) -> None:
        """Create receiver keys and write encrypted message to output file."""
        self.parent_path = Path(__file__).parent
        self.config: FileClient = {
            "path": "",
            "output": str(self.parent_path / "test.json"),
        }
        self.args = argparse.Namespace()
        self.args.receiver = HumanRSA()
        self.args.receiver.generate()
        self.args.date = "2022-01-01 00:00:00"

        msg = {"time": "2022-01-01 00:00:00"}
        signed_msg = sign_data(msg, self.args.receiver)
        ciphertext = encrypt_data(signed_msg, self.args.receiver)
        ciphertext = decode_data(ciphertext)
        self.encrypted_msg = json.dumps(ciphertext)
        with Path(self.config["output"]).open("w") as f:
            json.dump(ciphertext, f)

    def teardown_class(self) -> None:
        """Remove the temporary output JSON file."""
        output_path = Path(self.config["output"])
        if output_path.exists():
            output_path.unlink()

    def test_verify_mqtt_message(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Processing an encrypted MQTT message logs the decrypted payload."""
        obj = mqtt.Client()
        msg = mqtt.MQTTMessage()
        msg.payload = self.encrypted_msg.encode("latin-1")
        with caplog.at_level(logging.INFO, logger="consumer"):
            on_message(obj, self.args, msg)
        assert (
            re.search(
                (
                    r"Received message at 1970-01-01 \d{2}:\d{2}:\d{2}"
                    r"[+\-]\d{2}:\d{2}: "
                    r'{"time": "2022-01-01 00:00:00"}'
                ),
                caplog.text,
            )
            is not None
        )

    def test_verify_file_message(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifying an encrypted file logs timestamp and strips signature."""
        with caplog.at_level(logging.INFO, logger="consumer"):
            query_file(self.config, receiver=self.args.receiver)
        assert "2022, 1, 1, 0, 0" in caplog.text
        assert "signature" not in caplog.text


class TestConsumerPlaintext:
    """The consumer must work without encryption configured."""

    def test_query_file_no_receiver_logs_plaintext_item(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A plaintext output file is queried without any key configured."""
        output = tmp_path / "out.json"
        output.write_text(json.dumps({"time": "2022-01-01 00:00:00"}) + "\n")
        config: FileClient = {"path": "", "output": str(output)}

        with caplog.at_level(logging.INFO, logger="consumer"):
            query_file(config)

        assert "2022, 1, 1, 0, 0" in caplog.text

    def test_query_file_receiver_none_skips_decryption(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An explicit receiver=None must not attempt decryption."""
        output = tmp_path / "out.json"
        output.write_text(json.dumps({"time": "2022-01-01 00:00:00"}) + "\n")
        config: FileClient = {"path": "", "output": str(output)}

        with caplog.at_level(logging.INFO, logger="consumer"):
            query_file(config, receiver=None)

        assert "2022, 1, 1, 0, 0" in caplog.text

    def test_query_file_mixed_lines_decrypts_only_signed_items(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Only items carrying a signature field are decrypted."""
        output = tmp_path / "out.json"
        lines = [
            {"time": "2022-01-01 00:00:00"},
            {"time": "2022-01-01 00:00:01", "signature": "sig"},
        ]
        output.write_text(
            "\n".join(json.dumps(x) for x in lines) + "\n",
        )
        decrypt = MagicMock(
            return_value={"time": "2022-01-01 00:00:02"},
        )
        monkeypatch.setattr("consumer.verify_and_decrypt_data", decrypt)
        receiver = HumanRSA()
        receiver.generate()

        with caplog.at_level(logging.INFO, logger="consumer"):
            query_file(
                {"path": "", "output": str(output)},
                receiver=receiver,
            )

        decrypt.assert_called_once()
        assert decrypt.call_args[0][0]["signature"] == "sig"
        assert "2022, 1, 1, 0, 0" in caplog.text

    def test_on_message_receiver_none_logs_plaintext(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A plaintext MQTT message is logged when no key is configured."""
        userdata = argparse.Namespace()
        userdata.receiver = None
        msg = mqtt.MQTTMessage()
        msg.payload = b'{"time": "2022-01-01 00:00:00"}'

        with caplog.at_level(logging.INFO, logger="consumer"):
            on_message(mqtt.Client(), userdata, msg)

        assert '{"time": "2022-01-01 00:00:00"}' in caplog.text


class TestModelPresistence:
    """Tests for saving and loading models to/from disk."""

    def setup_class(self) -> None:
        """Initialise path and topic list used across persistence tests."""
        self.parent_path = Path(__file__).parent
        self.path = str(Path(__file__).parent / ".recovery_models/")
        self.topics = ["test"]

    def teardown_class(self) -> None:
        """Delete saved model pickles and remove the recovery directory."""
        models = list(
            Path(self.path).glob(
                f"model_{common_prefix(self.topics)}_*.pkl",
            ),
        )
        for model in models:
            model.unlink()
        Path(self.path).rmdir()

    def test_load_model(self) -> None:
        """Loading from a directory with no matching pickles returns None."""
        model = load_model(self.path, self.topics)
        assert model is None

    def test_save_model(self) -> None:
        """Saving writes one pickle; reloading returns equal object."""
        model = {"model": 1}
        save_model(self.path, self.topics, model)
        models = list(
            Path(self.path).glob(
                f"model_{common_prefix(self.topics)}_*.pkl",
            ),
        )
        assert len(models) == 1

        assert model == load_model(self.path, self.topics)
        assert load_model(self.path, ["bad_topics"]) is None

    def test_save_model_many_files_prunes_to_keep_last(
        self,
        tmp_path: Path,
    ) -> None:
        """Saving keeps only the newest keep_last recovery pickles."""
        prefix = f"model_{common_prefix(self.topics)}"
        for i in range(6):
            (tmp_path / f"{prefix}_20240101-00000{i}.pkl").touch()

        save_model(str(tmp_path), self.topics, {"model": 1}, keep_last=3)

        remaining = sorted(tmp_path.glob(f"{prefix}_*.pkl"))
        assert len(remaining) == 3
        # The two newest pre-existing files plus the just-saved one.
        names = [p.name for p in remaining]
        assert f"{prefix}_20240101-000004.pkl" in names
        assert f"{prefix}_20240101-000005.pkl" in names

    def test_save_model_keep_last_zero_disables_pruning(
        self,
        tmp_path: Path,
    ) -> None:
        """A non-positive keep_last leaves every recovery file alone."""
        prefix = f"model_{common_prefix(self.topics)}"
        for i in range(3):
            (tmp_path / f"{prefix}_20240101-00000{i}.pkl").touch()

        save_model(str(tmp_path), self.topics, {"model": 1}, keep_last=0)

        assert len(list(tmp_path.glob(f"{prefix}_*.pkl"))) == 4
