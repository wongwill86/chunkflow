import torch
import importlib
import types
import numpy as np
from chunkflow.chunk_operations.chunk_operation import ChunkOperation
from chunkblocks.global_offset_array import GlobalOffsetArray

def load_source(fname, module_name="Model"):
    """ Imports a module from source """
    loader = importlib.machinery.SourceFileLoader(module_name,fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod

class PytorchPatchInferenceEngine(ChunkOperation):
    def __init__(self, output_channels=1, output_datatype=None, model_file_name='./mito0.py',
                 weight_file_name='./mito0_220k.chkpt', use_bn=True, is_static_batch_norm=False ):
        super().__init__()

        self.net = load_source(model_file_name).InstantiatedModel
        self.net.load_state_dict(torch.load(weight_file_name, map_location=lambda location, storage: location))
        # self.net.cuda()
        self.net = torch.nn.DataParallel(self.net)

        if use_bn and is_static_batch_norm:
            self.net.eval()

        self.output_data_type = output_datatype
        self.output_channels = output_channels

    def run(self, patch):
        # patch should be a 5d np array
        #assert isinstance(patch, np.ndarray)
        if patch.ndim == 3:
            patch = patch.reshape((1, 1)+patch.shape)
        elif patch.ndim == 4:
            patch = patch.reshape((1, )+patch.shape)

        with torch.no_grad():
            in_v = torch.from_numpy(patch)#.cuda()
            # this net returns a list, but has one output
            output_v = self.net(in_v)[0]
            # the network output do not have sigmoid function
            output_patch = torch.sigmoid(output_v).data.cpu().numpy()
            return output_patch

    def _process(self, chunk):
        squeezed_data = self.run(chunk.data.astype(self.output_data_type)).squeeze()
        new_data = np.tile(squeezed_data, (self.output_channels,) + (1,) * len(squeezed_data.shape))
        chunk.data = GlobalOffsetArray(new_data, global_offset=(0,) + chunk.offset)
