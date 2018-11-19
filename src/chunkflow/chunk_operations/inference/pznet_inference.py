from chunkflow.chunk_operations.inference_operation import CachedNetworkReshapedInference

CACHED_NETS = {}

try:
    import pznet
except ImportError:
    PZNetInference = CachedNetworkReshapedInference
    pass


class PZNetInference(CachedNetworkReshapedInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_net(self):
        # "/nets/pinky100/unet4-long/mip1/cores2"
        net = pznet.znet(self.model_path, self.checkpoint_path)
        return net

    def _run(self, net, patch):
        return net.forward(patch)
