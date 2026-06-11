"""Minimal Pulsar producer example that emits three integer messages."""

from streamz import Stream

source = Stream()
producer_ = source.to_pulsar("pulsar://localhost:6650", "my-topic")

for i in range(3):
    source.emit(f"{i}".encode())

producer_.stop()
producer_.flush()
