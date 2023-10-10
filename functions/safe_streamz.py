import logging

from streamz import Stream

logger = logging.getLogger(__name__)


@Stream.register_api()
class map(Stream):
    def __init__(self, upstream, func, *args, **kwargs):
        self.func = func
        # this is one of a few stream specific kwargs
        stream_name = kwargs.pop('stream_name', None)
        self.kwargs = kwargs
        self.args = args

        Stream.__init__(self, upstream, stream_name=stream_name)

    def update(self, x, who=None, metadata=None):
        try:
            result = self.func(x, *self.args, **self.kwargs)
        except Exception as e:
            self.stop()
            self.destroy()
            logger.exception(e)
            raise e
        else:
            return self._emit(result, metadata=metadata)
