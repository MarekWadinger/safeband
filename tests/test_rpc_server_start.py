"""Tests for the configuration validation in RpcOutlierDetector.start."""

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import IO, Any, ClassVar

import pytest
from pandas import Timedelta
from river import proba, utils
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))

from functions.anomaly import GaussianScorer
from functions.model_persistence import load_model, save_model
from functions.typing_extras import (
    EmailConfig,
    FileClient,
    IOConfig,
    ModelConfig,
    MQTTClient,
    SetupConfig,
)
from rpc_server import RpcOutlierDetector, expand_model_params


class TestExpandModelParamsPhysicalLimits:
    """physical_limits config entries parse into per-signal bounds."""

    BASE: ClassVar[dict[str, Any]] = {
        "threshold": 0.99735,
        "t_e": Timedelta("1d"),
        "t_a": None,
        "t_g": None,
    }

    def test_absent_yields_none(self) -> None:
        """A configuration without physical_limits parses to None."""
        *_, limits = expand_model_params(ModelConfig(**self.BASE))
        assert limits is None

    def test_json_string_parses_to_bounds(self) -> None:
        """A JSON object string maps signal names to (low, high) pairs."""
        *_, limits = expand_model_params(
            ModelConfig(
                **self.BASE,
                physical_limits='{"plant/a": [0.0, 100.0]}',
            ),
        )
        assert limits == {"plant/a": (0.0, 100.0)}

    def test_mapping_passes_through_coerced(self) -> None:
        """An already-built mapping is coerced to float tuples."""
        *_, limits = expand_model_params(
            ModelConfig(**self.BASE, physical_limits={"plant/a": (0, 100)}),
        )
        assert limits == {"plant/a": (0.0, 100.0)}

    def test_non_mapping_raises(self) -> None:
        """A JSON value that is not an object is rejected."""
        with pytest.raises(TypeError, match="physical_limits"):
            expand_model_params(
                ModelConfig(**self.BASE, physical_limits="[0.0, 100.0]"),
            )

    def test_wrong_arity_raises(self) -> None:
        """Bounds that are not a (low, high) pair are rejected."""
        with pytest.raises(ValueError, match="physical_limits"):
            expand_model_params(
                ModelConfig(**self.BASE, physical_limits='{"plant/a": [0.0]}'),
            )


class TestStartWiresPhysicalLimits:
    """physical_limits from the service config reach the built model."""

    def test_start_univariate_model_receives_bounds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A single-topic model gets the bounds of its own signal."""
        recovery = tmp_path / "recovery"
        monkeypatch.setattr(
            RpcOutlierDetector,
            "run",
            lambda *_args, **_kwargs: None,
        )

        RpcOutlierDetector().start(
            FileClient(path="unused.csv", output=str(tmp_path / "out.json")),
            io=IOConfig(in_topics=["plant/a"], out_topics=None),
            model_params=ModelConfig(
                threshold=0.99735,
                t_e=Timedelta("1d"),
                t_a=Timedelta("1d"),
                t_g=Timedelta("1d"),
                physical_limits='{"plant/a": [0.0, 100.0]}',
            ),
            setup=SetupConfig(debug=True, recovery_path=str(recovery)),
        )

        model = load_model(str(recovery), ["plant/a"])
        assert model is not None
        assert model.physical_limits == (0.0, 100.0)

    def test_start_recovered_limits_mismatch_warns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A recovered model without bounds warns when the config has them."""
        recovery = tmp_path / "recovery"
        day = dt.timedelta(days=1)
        model = GaussianScorer(
            utils.TimeRolling(proba.Gaussian(), period=day),
            threshold=0.99735,
            grace_period=day,
            t_a=day,
        )
        save_model(str(recovery), ["plant/a"], model)
        monkeypatch.setattr(
            RpcOutlierDetector,
            "run",
            lambda *_args, **_kwargs: None,
        )

        with caplog.at_level(logging.WARNING, logger="rpc_server"):
            RpcOutlierDetector().start(
                FileClient(
                    path="unused.csv",
                    output=str(tmp_path / "out.json"),
                ),
                io=IOConfig(in_topics=["plant/a"], out_topics=None),
                model_params=ModelConfig(
                    threshold=0.99735,
                    t_e=Timedelta("1d"),
                    t_a=Timedelta("1d"),
                    t_g=Timedelta("1d"),
                    physical_limits='{"plant/a": [0.0, 100.0]}',
                ),
                setup=SetupConfig(debug=True, recovery_path=str(recovery)),
            )

        assert "physical_limits" in caplog.text
        assert "differs from configured" in caplog.text


class TestStartDebugGuard:
    """Debug mode is only valid together with a file client."""

    def test_debug_with_remote_broker_raises(self) -> None:
        """Combining debug mode with an MQTT broker config is rejected."""
        with pytest.raises(ValueError, match="requires a file client"):
            RpcOutlierDetector().start(
                MQTTClient(host="broker", port=1883),
                io=IOConfig(in_topics=["plant/a"], out_topics=None),
                model_params=ModelConfig(
                    threshold=0.99735,
                    t_e=Timedelta("1d"),
                    t_a=None,
                    t_g=None,
                ),
                setup=SetupConfig(debug=True),
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
            FileClient(path="unused.csv", output=str(output)),
            io=IOConfig(in_topics=["plant/a"], out_topics=None),
            model_params=ModelConfig(
                threshold=0.99735,
                t_e=Timedelta("1d"),
                t_a=Timedelta("1d"),
                t_g=Timedelta("1d"),
            ),
            setup=SetupConfig(debug=True),
        )

        lines = output.read_text().strip().splitlines()
        assert len(lines) == 1
        assert "anomaly" in json.loads(lines[0])


class TestStartEmailWithEncryption:
    """Email alerting must see plaintext results even when encrypting."""

    def test_start_email_branch_receives_plaintext_when_encrypted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With key_path set, the email sink gets unencrypted records."""
        received: list[tuple[dict, ...]] = []

        def capture(
            _self: RpcOutlierDetector,
            xs: tuple[dict, ...],
            _email_client: object,
            _model: object,
        ) -> None:
            received.append(xs)

        def fake_run(
            _self: RpcOutlierDetector,
            _config: object,
            source: Stream,
            _detector: Stream,
            _debug: bool,
        ) -> None:
            source.emit(b"21.0")
            source.emit(b"22.0")

        monkeypatch.setattr(RpcOutlierDetector, "send_anomaly_email", capture)
        monkeypatch.setattr(RpcOutlierDetector, "run", fake_run)

        RpcOutlierDetector().start(
            FileClient(path="unused.csv", output=str(tmp_path / "out.json")),
            io=IOConfig(in_topics=["plant/a"], out_topics=None),
            model_params=ModelConfig(
                threshold=0.99735,
                t_e=Timedelta("1d"),
                t_a=Timedelta("1d"),
                t_g=Timedelta("1d"),
            ),
            setup=SetupConfig(debug=True, key_path=str(tmp_path / "keys")),
            email=EmailConfig(
                sender_email="sender@example.com",
                sender_password="secret",  # noqa: S106
                recipient_email="ops@example.com",
            ),
        )

        assert received
        window = received[-1]
        assert len(window) == 2
        for x in window:
            # Encrypted records carry string payloads; the email branch
            # must compute on plaintext anomaly flags.
            assert isinstance(x["anomaly"], int)


class TestStartRecoveredModelParams:
    """A recovered model must not silently ignore the current config."""

    @staticmethod
    def _save_recovery_model(recovery: Path, threshold: float) -> None:
        day = dt.timedelta(days=1)
        model = GaussianScorer(
            utils.TimeRolling(proba.Gaussian(), period=day),
            threshold=threshold,
            grace_period=day,
            t_a=day,
        )
        save_model(str(recovery), ["plant/a"], model)

    def _start(self, tmp_path: Path, recovery: Path) -> None:
        RpcOutlierDetector().start(
            FileClient(path="unused.csv", output=str(tmp_path / "out.json")),
            io=IOConfig(in_topics=["plant/a"], out_topics=None),
            model_params=ModelConfig(
                threshold=0.99735,
                t_e=Timedelta("1d"),
                t_a=Timedelta("1d"),
                t_g=Timedelta("1d"),
            ),
            setup=SetupConfig(debug=True, recovery_path=str(recovery)),
        )

    def test_start_recovered_params_mismatch_warns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A threshold differing from the config is reported loudly."""
        recovery = tmp_path / "recovery"
        self._save_recovery_model(recovery, threshold=0.5)
        monkeypatch.setattr(
            RpcOutlierDetector,
            "run",
            lambda *_args, **_kwargs: None,
        )

        with caplog.at_level(logging.WARNING, logger="rpc_server"):
            self._start(tmp_path, recovery)

        assert "threshold" in caplog.text
        assert "differs from configured" in caplog.text

    def test_start_recovered_params_match_no_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A recovered model matching the config stays silent."""
        recovery = tmp_path / "recovery"
        self._save_recovery_model(recovery, threshold=0.99735)
        monkeypatch.setattr(
            RpcOutlierDetector,
            "run",
            lambda *_args, **_kwargs: None,
        )

        with caplog.at_level(logging.WARNING, logger="rpc_server"):
            self._start(tmp_path, recovery)

        assert "differs from configured" not in caplog.text


class TestStartClosesFiles:
    """Output files opened by the sink are closed on any shutdown path."""

    def test_start_run_raises_output_file_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A crash in run() still flushes and closes the output file."""
        opened: list[IO[Any]] = []
        orig_open = Path.open

        def spy_open(
            self: Path,
            *args: object,
            **kwargs: object,
        ) -> IO[Any]:
            # The spy only records the handle; args pass through verbatim.
            f = orig_open(self, *args, **kwargs)  # type: ignore
            opened.append(f)
            return f

        def boom(
            _self: RpcOutlierDetector,
            _config: object,
            _source: Stream,
            _detector: Stream,
            _debug: bool,
        ) -> None:
            msg = "boom"
            raise RuntimeError(msg)

        monkeypatch.setattr(Path, "open", spy_open)
        monkeypatch.setattr(RpcOutlierDetector, "run", boom)

        with pytest.raises(RuntimeError, match="boom"):
            RpcOutlierDetector().start(
                FileClient(
                    path="unused.csv",
                    output=str(tmp_path / "out.json"),
                ),
                io=IOConfig(in_topics=["plant/a"], out_topics=None),
                model_params=ModelConfig(
                    threshold=0.99735,
                    t_e=Timedelta("1d"),
                    t_a=Timedelta("1d"),
                    t_g=Timedelta("1d"),
                ),
                setup=SetupConfig(debug=True),
            )

        assert opened
        assert all(f.closed for f in opened)
