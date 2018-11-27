from concurrent.futures import ProcessPoolExecutor


class ChunkOperation:
    def _process(self, chunk):
        """
        returns processed chunk
        """
        raise NotImplementedError

    def __call__(self, chunk):
        self._process(chunk)
        return chunk


class OffProcessChunkOperation(ChunkOperation):
    def __init__(self, operation, parallelism=1):
        self.operation = operation
        self.pool = ProcessPoolExecutor(max_workers=parallelism)

    def __call__(self, chunk):
        # return self.pool.submit(create_operation_and_execute, self.operation_class, self.args, self.kwargs, chunk)
        return self.pool.submit(self.operation.__call__, chunk)
