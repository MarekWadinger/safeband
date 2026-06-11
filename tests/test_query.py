"""Tests for MQTT message consumption and model persistence utilities."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

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
        """Create receiver keys and write an encrypted message to the output file."""
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
        self, caplog: pytest.LogCaptureFixture
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
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Reading and verifying an encrypted file logs the message timestamp and strips the signature."""
        with caplog.at_level(logging.INFO, logger="consumer"):
            query_file(self.config, receiver=self.args.receiver)
        assert "2022, 1, 1, 0, 0" in caplog.text
        assert "signature" not in caplog.text


class TestModelPresistence:
    """Tests for saving and loading models to/from disk."""

    def setup_class(self) -> None:
        """Initialise path and topic list used across persistence tests."""
        self.parent_path = Path(__file__).parent
        self.path = str(Path(__file__).parent / ".recovery_models/")
        self.topics = ["test"]

    def teardown_class(self) -> None:
        """Delete saved model pickle files and remove the recovery directory."""
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
        """Saving a model writes one pickle; reloading returns an equal object; unknown topics return None."""
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
