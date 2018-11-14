import collections
import importlib
import types

import numpy as np

from chunkblocks.global_offset_array import GlobalOffsetArray

from chunkflow.chunk_operations.inference_operation import InferenceOperation

CACHED_NETS = {}

try:
    import torch
except ImportError:
    PyTorchInference = InferenceOperation
    pass


def load_source(fname, module_name="Model"):
    """ Imports a module from source """
    loader = importlib.machinery.SourceFileLoader(module_name, fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


class PyTorchInference(InferenceOperation):
    def __init__(self, patch_shape,
                 model_location='./models/mito0.py',
                 checkpoint_location='./models/mito0_220k.chkpt',
                 gpu=False,
                 accelerator_ids=None,
                 use_bn=True, is_static_batch_norm=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_patch_shape = (1,) + patch_shape
        self.model_location = model_location
        self.checkpoint_location = checkpoint_location
        self.gpu = gpu
        self.accelerator_ids = accelerator_ids
        self.use_bn = use_bn
        self.is_static_batch_norm = is_static_batch_norm
        self.key = (self.channel_patch_shape, self.model_location, self.checkpoint_location,
                    self.gpu, tuple(self.accelerator_ids), self.use_bn, self.is_static_batch_norm)

    def get_or_create_net(self):
        if self.key in CACHED_NETS:
            return CACHED_NETS[self.key]

        in_spec = dict(input=self.channel_patch_shape)
        out_spec = collections.OrderedDict(mito=self.channel_patch_shape)

        model = load_source(self.model_location).Model(in_spec, out_spec)

        if self.gpu:
            checkpoint = torch.load(self.checkpoint_location)
            model.load_state_dict(checkpoint)
            model.cuda()
            net = torch.nn.DataParallel(model, device_ids=self.accelerator_ids)
        else:
            checkpoint = torch.load(self.checkpoint_location, map_location=lambda location, storage: location)
            model.load_state_dict(checkpoint)
            net = torch.nn.DataParallel(model)

        if self.use_bn and self.is_static_batch_norm:
            net.eval()

        CACHED_NETS[self.key] = net

        return net

    def run(self, patch):
        net = self.get_or_create_net()

        # patch should be a 5d np array
        patch = patch.reshape((1,) * (5 - patch.ndim) + patch.shape)

        with torch.no_grad():
            in_v = torch.from_numpy(patch)
            if self.gpu:
                in_v = in_v.cuda()

            # this net returns a list, but has one output
            output_v = net(in_v)[0]

            # the network output does not have a sigmoid function
            output_patch = torch.sigmoid(output_v).data.cpu().numpy()
            return output_patch

    def _process(self, chunk):
        output = self.run(chunk.data.astype(self.output_datatype))
        if output.shape[0] < self.output_channels:
            squeezed_output = output.squeeze()
            output = np.tile(squeezed_output, (self.output_channels,) + (1,) * len(squeezed_output.shape))
        chunk.data = GlobalOffsetArray(output, global_offset=(0,) + chunk.offset)
