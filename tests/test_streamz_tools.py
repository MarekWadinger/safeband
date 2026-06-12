"""Tests for the custom map stream operator error handling."""

import logging
import sys
from pathlib import Path

import pytest
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))

# Importing registers the custom ``map`` operator on Stream.
import functions.streamz_tools  # noqa: F401


def _reciprocal(x: int) -> float:
    return 1 / x


class TestMapStreamOnError:
    """Error semantics of the registered map operator."""

    def test_map_update_func_raises_skips_message(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failing message is logged and dropped; the next one flows."""
        results: list[float] = []
        source = Stream()
        source.map(_reciprocal).sink(results.append)

        with caplog.at_level(
            logging.ERROR,
            logger="functions.streamz_tools",
        ):
            source.emit(0)
        source.emit(1)

        assert results == [1.0]
        assert "dropping message" in caplog.text

    def test_map_update_on_error_raise_propagates(self) -> None:
        """With on_error='raise', the exception stops the stream."""
        source = Stream()
        # Hold a reference: streamz tracks downstream nodes weakly.
        mapped = source.map(_reciprocal, on_error="raise")

        with pytest.raises(ZeroDivisionError):
            source.emit(0)
        assert mapped.upstreams == []

    def test_map_init_invalid_on_error_raises_valueerror(self) -> None:
        """An unsupported on_error value is rejected at wiring time."""
        with pytest.raises(ValueError, match="on_error"):
            Stream().map(_reciprocal, on_error="bogus")
