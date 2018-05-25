class ChunkOperation(object):
    def _process(self, chunk):
        """
        returns processed data
        """
        raise NotImplemented

    def __call__(self, chunk):
        self._process(chunk)
        return chunk
