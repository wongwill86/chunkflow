import sys

from chunkflow.chunk_operations.inference_operation import CachedNetworkReshapedInference


class PZNetInference(CachedNetworkReshapedInference):
    def _create_net(self):
        sys.path.append(self.model_path)
        import pznet
        net = pznet.znet().load_net(self.model_path)
        return net

    def _run(self, net, patch):
        return net.forward(patch)
