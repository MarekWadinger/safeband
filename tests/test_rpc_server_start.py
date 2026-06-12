"""Tests for the configuration validation in RpcOutlierDetector.start."""

import json
import sys
from pathlib import Path

import pytest
from pandas import Timedelta
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))

import rpc_server
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


class TestStartSkipsBadMessages:
    """One malformed message must not take down the pipeline."""

    def test_start_junk_payload_skipped_valid_processed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A junk payload is dropped and the next valid one is sinked."""

        def fake_run(
            _self: RpcOutlierDetector,
            _config: object,
            source: Stream,
            _detector: Stream,
            _debug: bool,
        ) -> None:
            source.emit(b"junk")
            source.emit(b"21.0")

        monkeypatch.setattr(RpcOutlierDetector, "run", fake_run)
        output = tmp_path / "out.json"

        RpcOutlierDetector().start(
            {"path": "unused.csv", "output": str(output)},
            io={"in_topics": ["plant/a"], "out_topics": None},
            model_params={
                "threshold": 0.99735,
                "t_e": Timedelta("1d"),
                "t_a": Timedelta("1d"),
                "t_g": Timedelta("1d"),
            },
            setup={"debug": True},
        )
        rpc_server._exit_stack.close()

        lines = output.read_text().strip().splitlines()
        assert len(lines) == 1
        assert "anomaly" in json.loads(lines[0])
