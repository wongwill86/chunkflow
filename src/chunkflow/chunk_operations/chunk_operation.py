class ChunkOperation(object):
    def _process(self, chunk):
        """
        returns processed data
        """
        raise NotImplementedError

    def __call__(self, chunk):
        self._process(chunk)
        return chunk
