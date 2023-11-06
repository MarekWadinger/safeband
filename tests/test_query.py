import argparse
import glob
import json
import os
import re
import sys
from io import StringIO
from pathlib import Path

import paho.mqtt.client as mqtt
from human_security import HumanRSA

sys.path.insert(1, str(Path(__file__).parent.parent))
from consumer import on_message, query_file  # noqa: E402
from functions.encryption import (  # noqa: E402
    decode_data,
    encrypt_data,
    sign_data,
)
from functions.model_persistence import load_model, save_model  # noqa: E402
from functions.typing_extras import FileClient  # noqa: E402
from functions.utils import common_prefix  # noqa: E402


class TestConsumer:
    def setup_class(self):
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
        self.encrypted_msg = json.dumps(ciphertext.copy())
        with open(self.config["output"], "w") as f:
            json.dump(ciphertext, f)

    def teardown_class(self):
        if os.path.exists(self.config["output"]):
            os.remove(self.config["output"])

    def test_verify_mqtt_message(self):
        obj = mqtt.Client()
        msg = mqtt.MQTTMessage()
        msg.payload = self.encrypted_msg.encode("latin-1")
        stdout_ = sys.stdout  # Keep track of the previous value.
        f = StringIO()
        sys.stdout = f
        on_message(obj, self.args, msg)
        sys.stdout = stdout_  # restore the previous stdout.
        assert (
            re.match(
                (
                    r"Received message at 1970-01-01 \d{2}:\d{2}:\d{2}: "
                    r'{"time": "2022-01-01 00:00:00"}\n'
                ),
                f.getvalue(),
            )
            is not None
        )

    def test_verify_file_message(self):
        f = StringIO()
        stdout_ = sys.stdout  # Keep track of the previous value.
        sys.stdout = f
        query_file(self.config, **{"receiver": self.args.receiver})
        sys.stdout = stdout_
        assert (
            f.getvalue() == "{'time': datetime.datetime(2022, 1, 1, 0, 0)}\n"
        )


class TestModelPresistence:
    def setup_class(self):
        self.parent_path = Path(__file__).parent
        self.path = str(Path(__file__).parent / ".recovery_models/")
        self.topics = ["test"]

    def teardown_class(self):
        models = glob.glob(
            os.path.join(
                self.path, f"model_{common_prefix(self.topics)}_*.pkl"
            )
        )
        for model in models:
            os.remove(model)
        os.rmdir(self.path)

    def test_load_model(self):
        model = load_model(self.path, self.topics)
        assert model is None

    def test_save_model(self):
        model = {"model": 1}
        save_model(self.path, self.topics, model)
        models = glob.glob(
            os.path.join(
                self.path, f"model_{common_prefix(self.topics)}_*.pkl"
            )
        )
        assert len(models) == 1

        assert model == load_model(self.path, self.topics)
        assert load_model(self.path, ["bad_topics"]) is None
