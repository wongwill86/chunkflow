from concurrent.futures import ProcessPoolExecutor


def create_operation_and_execute(operation, chunk):
    return operation(chunk)


class ChunkOperation:
    def _process(self, chunk):
        """
        returns processed data
        """
        raise NotImplementedError

    def __call__(self, chunk):
        self._process(chunk)
        return chunk


class DeferredChunkOperation(ChunkOperation):
    def __init__(self, operation, parallelism=1, *args, **kwargs):
        self.operation = operation
        self.pool = ProcessPoolExecutor(max_workers=parallelism)

    def __call__(self, chunk):
        return self.pool.submit(self.operation.__call__, chunk)
