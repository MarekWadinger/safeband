from streamz import Stream

source = Stream()
producer_ = source.to_pulsar(
    'my-topic', producer_config={
        'service_url': 'pulsar://localhost:6650'})

for i in range(3):
    source.emit(('hello-pulsar-%d' % i).encode('utf-8'))

producer_.stop()
producer_.flush()
