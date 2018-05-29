import numpy as np

from chunkflow.chunk_operations.chunk_operation import ChunkOperation
from chunkflow.global_offset_array import GlobalOffsetArray


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

class IncrementThreeChannelInference(ChunkOperation):
    def __init__(self, step=1, *args, **kwargs):
        self.step = step
        super().__init__(*args, **kwargs)

    def _process(self, chunk):
        self.run_inference(chunk)

    def run_inference(self, chunk):
        chunk.data += self.step
        global_offset = (0,) + chunk.data.global_offset
        one = chunk.data
        two = chunk.data * 10
        three = chunk.data * 100
        chunk.data = GlobalOffsetArray(np.stack((one, two, three)), global_offset=global_offset)
