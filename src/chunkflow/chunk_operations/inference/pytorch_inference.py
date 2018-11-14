import collections
import torch
import importlib
import types
import numpy as np
from chunkflow.chunk_operations.inference_operation import InferenceOperation
from chunkflow.chunk_operations.chunk_operation import ChunkOperation
from chunkblocks.global_offset_array import GlobalOffsetArray

def load_source(fname, module_name="Model"):
    """ Imports a module from source """
    loader = importlib.machinery.SourceFileLoader(module_name,fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod

class PyTorchInference(InferenceOperation):
    def __init__(self, patch_shape,
                 model_location='./src/chunkflow/chunk_operations/inference/mito0.py',
                 checkpoint_location='./src/chunkflow/chunk_operations/inference/mito0_220k.chkpt',
                 gpu=False,
                 accelerator_ids=None,
                 use_bn=True, is_static_batch_norm=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        channel_patch_shape = (1,) + patch_shape
        in_spec = dict(input=channel_patch_shape)
        out_spec = collections.OrderedDict(mito=channel_patch_shape)

        model = load_source(model_location).Model(in_spec, out_spec)

        if gpu:
            checkpoint = torch.load(checkpoint_location)
            model.load_state_dict(checkpoint)
            model.cuda()
            self.net = torch.nn.DataParallel(model, device_ids=accelerator_ids)
        else:
            checkpoint = torch.load(checkpoint_location, map_location=lambda location, storage: location)
            model.load_state_dict(checkpoint)
            self.net = torch.nn.DataParallel(model)

        if use_bn and is_static_batch_norm:
            self.net.eval()
        self.gpu = gpu

    def run(self, patch):
        # patch should be a 5d np array
        patch = patch.reshape((1,) * (5 - patch.ndim) + patch.shape)

        with torch.no_grad():
            in_v = torch.from_numpy(patch)
            if self.gpu:
                in_v = in_v.cuda()

            # this net returns a list, but has one output
            output_v = self.net(in_v)[0]

            # the network output does not have a sigmoid function
            output_patch = torch.sigmoid(output_v).data.cpu().numpy()
            return output_patch

    def _process(self, chunk):
        output = self.run(chunk.data.astype(self.output_datatype))
        if output.shape[0] < self.output_channels:
            squeezed_output = output.squeeze()
            output = np.tile(squeezed_output, (self.output_channels,) + (1,) * len(squeezed_output.shape))
        chunk.data = GlobalOffsetArray(output, global_offset=(0,) + chunk.offset)
