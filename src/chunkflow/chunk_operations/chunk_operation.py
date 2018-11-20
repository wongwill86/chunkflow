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


class OffProcessChunkOperation(ChunkOperation):
    def __init__(self, operation_class, parallelism=1, *args, **kwargs):
        self.operation_class = operation_class
        self.args = args
        self.kwargs = kwargs
        self.pool = ProcessPoolExecutor(max_workers=parallelism)

    def __call__(self, chunk):
        operation = self.operation_class(*self.args, **self.kwargs)
        return self.pool.submit(operation.__call__, chunk)
