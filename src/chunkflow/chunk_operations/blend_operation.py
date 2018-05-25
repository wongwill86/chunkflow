from chunkflow.chunk_operations.chunk_operation import ChunkOperation


class IdentityBlend(ChunkOperation):
    def _process(self, chunk):
        pass

class NormalizedBlend(ChunkOperation):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def _process(self, chunk):
        self.run_blend(chunk)

    def run_blend(self, chunk):
        factor = len(chunk.size) ** len(chunk.size)
        print(factor)
        for overlap_slices in chunk.border_slices():
            chunk.data[overlap_slices] /= factor
