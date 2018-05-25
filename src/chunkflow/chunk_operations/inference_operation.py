from chunkflow.chunk_operations.chunk_operation import ChunkOperation


class IdentityInference(ChunkOperation):
    def _process(self, chunk):
        pass


class IncrementInference(ChunkOperation):
    def __init__(self, step=1, *args, **kwargs):
        self.step = step
        super().__init__(*args, **kwargs)

    def _process(self, chunk):
        self.run_inference(chunk)

    def run_inference(self, chunk):
        chunk.data += self.step
