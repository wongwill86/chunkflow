from chunkflow.chunk_operations.inference_operation import CachedNetworkReshapedInference
from chunkflow.io import load_source
import sys


class PZNetInference(CachedNetworkReshapedInference):
    def _create_net(self):
        net_directory = load_source(self.model_path)
        sys.path.append(net_directory)
        import pznet
        net = pznet.znet().load_net(net_directory)
        return net

    def _run(self, net, patch):
        return net.forward(patch)
