import numpy as np
from memory_profiler import profile

from chunkflow.chunk_operations.chunk_operation import ChunkOperation


class IdentityInference(ChunkOperation):
    def __init__(self, output_channels=1, output_data_type=None):
        self.output_data_type = output_data_type
        self.output_channels = output_channels

    @profile
    def _process(self, chunk):
        if self.output_data_type is not None:
            # old_data = chunk.data
            # del chunk.data
            # chunk.data = old_data.astype(self.output_data_type)
            chunk.data = chunk.data.astype(self.output_data_type)

        if self.output_channels > 1:
            squeezed_data = chunk.data.squeeze()
            new_data = np.tile(squeezed_data, (self.output_channels,) + (1,) * len(squeezed_data.shape))
            chunk.data = new_data
            # del squeezed_data


class InferenceFactory:
    def __init__(self, patch_shape, output_channels=3, output_data_type=None, model_path=None, net_path=None,
                 accelerator_ids=None):
        self.patch_shape = patch_shape
        self.output_channels = output_channels
        self.output_data_type = output_data_type

    def get_operation(self, framework, model_path, net_path, accelerator_ids):
        if framework == 'identity':
            return IdentityInference(self.output_channels, self.output_data_type)
        else:
            return IdentityInference(self.output_channels, self.output_data_type)
