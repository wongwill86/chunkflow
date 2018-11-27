import collections

from chunkflow.chunk_operations.inference_operation import CachedNetworkInference
from chunkflow.io import load_source


class PyTorchInference(CachedNetworkInference):
    def _create_net(self):
        import torch
        in_spec = dict(input=self.channel_patch_shape)
        out_spec = collections.OrderedDict(out=self.channel_patch_shape)

        model = load_source(self.model_path).Model(in_spec, out_spec)

        if self.gpu:
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint)
            model.cuda()
            net = torch.nn.DataParallel(model, device_ids=self.accelerator_ids)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location=lambda location, storage: location)
            model.load_state_dict(checkpoint)
            net = torch.nn.DataParallel(model)

        if self.use_bn and self.is_static_batch_norm:
            net.eval()

        return net

    def _run(self, net, patch):
        import torch
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
