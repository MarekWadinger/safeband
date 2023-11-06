from streamz import Stream

source = Stream()
producer_ = source.to_pulsar("pulsar://localhost:6650", "my-topic")

for i in range(3):
    source.emit(("%d" % i).encode("utf-8"))

producer_.stop()
producer_.flush()
