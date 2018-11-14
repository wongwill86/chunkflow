import numpy as np

from chunkflow.chunk_operations.chunk_operation import ChunkOperation, DeferredChunkOperation


class InferenceOperation(ChunkOperation):
    def __init__(self, output_channels=1, output_datatype=None, *args, **kwargs):
        self.output_channels = output_channels
        self.output_datatype = output_datatype
        super().__init__(*args, **kwargs)

    def _process(self, Chunk):
        raise NotImplementedError


class IdentityInferenceOperation(InferenceOperation):
    def _process(self, chunk):
        if self.output_datatype is not None:
            chunk.data = chunk.data.astype(self.output_datatype)

        if self.output_channels > 1:
            squeezed_data = chunk.data.squeeze()
            new_data = np.tile(squeezed_data, (self.output_channels,) + (1,) * len(squeezed_data.shape))
            chunk.data = new_data
            # del squeezed_data


class InferenceFactory:
    def __init__(self, patch_shape, output_channels=1, output_datatype=None, gpu=False, accelerator_ids=None):
        self.patch_shape = patch_shape
        self.output_channels = output_channels
        self.output_datatype = output_datatype
        self.gpu = gpu
        self.accelerator_ids = accelerator_ids

    def get_operation(self, framework, model_path, checkpoint_path, deferred_processing=False, parallelism=1):
        if framework == 'identity':
            operation = IdentityInferenceOperation(self.output_channels, self.output_datatype)
        elif framework == 'pytorch':
            from chunkflow.chunk_operations.inference.pytorch_inference import PyTorchInference
            operation = PyTorchInference(self.patch_shape, output_channels=self.output_channels,
                                         output_datatype=self.output_datatype, gpu=self.gpu,
                                         accelerator_ids=self.accelerator_ids)
        else:
            operation = IdentityInferenceOperation(self.output_channels, self.output_datatype)

        if deferred_processing:
            return DeferredChunkOperation(operation, parallelism=parallelism)
        else:
            return operation
