"""Tests for the configuration validation in RpcOutlierDetector.start."""

import sys
from pathlib import Path

import pytest
from pandas import Timedelta

sys.path.insert(1, str(Path(__file__).parent.parent))

from rpc_server import RpcOutlierDetector


class TestStartDebugGuard:
    """Debug mode is only valid together with a file client."""

    def test_debug_with_remote_broker_raises(self) -> None:
        """Combining debug mode with an MQTT broker config is rejected."""
        with pytest.raises(ValueError, match="requires a file client"):
            RpcOutlierDetector().start(
                {"host": "broker", "port": 1883},
                io={"in_topics": ["plant/a"], "out_topics": None},
                model_params={
                    "threshold": 0.99735,
                    "t_e": Timedelta("1d"),
                    "t_a": None,
                    "t_g": None,
                },
                setup={"debug": True},
            )
