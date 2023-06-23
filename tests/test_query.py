import argparse
from io import StringIO
import json
import os
import paho.mqtt.client as mqtt
import sys
from pathlib import Path

from human_security import HumanRSA

sys.path.insert(1, str(Path(__file__).parent.parent))
from functions.encryption import (  # noqa: E402
    encrypt_data, sign_data, decode_data)
from query_signal_limits import on_message, query_file  # noqa: E402


class TestSecurity():

    def setup_class(self):
        self.parent_path = Path(__file__).parent
        self.args = argparse.Namespace()
        self.args.receiver = HumanRSA()
        self.args.receiver.generate()
        self.args.broker = self.parent_path / 'test.json'
        self.args.date = "2022-01-01 00:00:00"

        msg = {'time': "2022-01-01 00:00:00"}
        signed_msg = sign_data(msg, self.args.receiver)
        ciphertext = encrypt_data(signed_msg, self.args.receiver)
        ciphertext = decode_data(ciphertext)
        self.encrypted_msg = json.dumps(ciphertext.copy())
        with open(self.args.broker, 'w') as f:
            json.dump(ciphertext, f)

    def teardown_class(self):
        if os.path.exists(self.args.broker):
            os.remove(self.args.broker)

    def test_verify_mqtt_message(self):
        obj = mqtt.Client()
        msg = mqtt.MQTTMessage()
        msg.payload = self.encrypted_msg.encode('latin-1')
        stdout_ = sys.stdout  # Keep track of the previous value.
        f = StringIO()
        sys.stdout = f
        on_message(obj, self.args, msg)
        sys.stdout = stdout_  # restore the previous stdout.
        assert (f.getvalue() ==
                'Received message: {"time": "2022-01-01 00:00:00"}\n')

    def test_verify_file_message(self):
        f = StringIO()
        stdout_ = sys.stdout  # Keep track of the previous value.
        sys.stdout = f
        query_file(self.args)
        sys.stdout = stdout_
        assert (f.getvalue() ==
                "{'time': datetime.datetime(2022, 1, 1, 0, 0)}\n")
