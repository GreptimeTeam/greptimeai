from openai._streaming import Stream, AsyncStream


class StreamUtil:
    @staticmethod
    def is_stream(obj):
        return obj and isinstance(obj, Stream)

    @staticmethod
    def is_async_stream(obj):
        return obj and isinstance(obj, AsyncStream)
