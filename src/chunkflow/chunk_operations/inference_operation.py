import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray

from chunkflow.chunk_operations.chunk_operation import ChunkOperation, OffProcessChunkOperation
from chunkflow.io import download_to_local

CACHED_NETS = {}


class InferenceOperation(ChunkOperation):
    def __init__(self, patch_shape, channel_dimensions=(1,), output_datatype=None,
                 model_path=None,  # 'file://~/src/chunkflow/models/mito0.py',
                 checkpoint_path=None,  # 'file://~/src/chunkflow/models/mito0_220k.chkpt',
                 gpu=False,
                 accelerator_ids=None,
                 use_bn=True, is_static_batch_norm=False, should_download=True):

        self.channel_dimensions = channel_dimensions
        self.output_datatype = output_datatype
        self.channel_patch_shape = (1,) + patch_shape
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        if model_path is not None and should_download:
            self.model_path = download_to_local(model_path)
        if checkpoint_path is not None and should_download:
            self.checkpoint_path = download_to_local(checkpoint_path)
        self.gpu = gpu
        self.accelerator_ids = accelerator_ids
        self.use_bn = use_bn
        self.is_static_batch_norm = is_static_batch_norm
        self.key = (self.channel_patch_shape, model_path, checkpoint_path, gpu,
                    tuple(accelerator_ids) if accelerator_ids is not None else None, use_bn, is_static_batch_norm)

    def _run(self, net, patch):
        raise NotImplementedError


class CachedNetworkInference(InferenceOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, should_download=False, **kwargs)
        if self.model_path is not None:
            self.model_path = download_to_local(self.model_path)
        if self.checkpoint_path is not None:
            self.checkpoint_path = download_to_local(self.checkpoint_path)

    def _create_net(self):
        raise NotImplementedError

    def get_or_create_net(self):
        if self.key in CACHED_NETS:
            return CACHED_NETS[self.key]
        else:
            print('Creating network, configuration:', self.key)
            net = self._create_net()
            CACHED_NETS[self.key] = net
            return net

    def _process(self, chunk):
        patch = chunk.data.astype(self.output_datatype)

        # Currently our networks are automatically assumed to be 5d
        # This function will have to change if there is more than 1 dimension for the output channel
        patch = patch.reshape((1,) * (5 - patch.ndim) + patch.shape)

        net = self.get_or_create_net()

        output = self._run(net, patch)

        output = output.reshape(self.channel_dimensions + chunk.shape)
        chunk.data = GlobalOffsetArray(output, global_offset=len(self.channel_dimensions) * (0,) + chunk.offset)


class IdentityInferenceOperation(InferenceOperation):
    def _process(self, chunk):
        if self.output_datatype is not None:
            chunk.data = chunk.data.astype(self.output_datatype)

        if self.channel_dimensions != (1,):
            squeezed_data = chunk.data.squeeze()
            new_data = np.tile(squeezed_data, self.channel_dimensions + (1,) * len(squeezed_data.shape))
            chunk.data = new_data


class InferenceFactory:
    def __init__(self, patch_shape, channel_dimensions=(1,), output_datatype=None, gpu=False, accelerator_ids=None):
        self.patch_shape = patch_shape
        self.channel_dimensions = channel_dimensions
        self.output_datatype = output_datatype
        self.gpu = gpu
        self.accelerator_ids = accelerator_ids

    def get_operation(self, framework, model_path, checkpoint_path, off_main_process=False, parallelism=1):
        if framework == 'identity':
            inference_operation_class = IdentityInferenceOperation
        elif framework == 'pytorch':
            from chunkflow.chunk_operations.inference.pytorch_inference import PyTorchInference
            inference_operation_class = PyTorchInference
        elif framework == 'pznet':
            from chunkflow.chunk_operations.inference.pznet_inference import PZNetInference
            inference_operation_class = PZNetInference
        else:
            inference_operation_class = IdentityInferenceOperation

        inference_operation = inference_operation_class(self.patch_shape,
                                                        model_path=model_path, checkpoint_path=checkpoint_path,
                                                        channel_dimensions=self.channel_dimensions,
                                                        output_datatype=self.output_datatype, gpu=self.gpu)

        if off_main_process:
            return OffProcessChunkOperation(inference_operation, parallelism=parallelism)
        else:
            return inference_operation
