"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mchunkflow` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``chunkflow.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``chunkflow.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import click
import ast

# https://stackoverflow.com/a/47730333
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

@click.group()
# @click.option('--output_chunk_start', type=int, help="the start coordinates of final output block",
#               cls=PythonLiteralOption, required=True)
# @click.option('--output_chunk_size', type=int, help="the size of output block",
#               cls=PythonLiteralOption, default=[112, 1152, 1152])
# @click.option('--overlap', type=int, help="overlap by number of voxels",
#               cls=PythonLiteralOption, default=[4, 64, 64])
# @click.option('--patch_size', type=int, help="convnet input patch size",
#               cls=PythonLiteralOption, default=[32, 256, 256])
# # @click.option('--no_eval', action='store_true', help="this is on then using dynamic \ batchnorm, otherwise static.")
# # @click.option('--output_key', type=str, default='affinity', help="the name of the final output layer")
def main():
    """
    Set up configuration
    """
    print('hi')


# @main.command()
# # @click.option('--input_image_source', type=str, help="input image source path, i.e. file://, gs://, or s3://.",
# #               required=True)
# # @click.option('--output_core_destination', type=str, help="destination path for the valid core output of the chunk,\
# #               prefixes supported: file://, gs://, s3://.",
# #               required=True)
# # @click.option('--output_overlap_destination', type=str, help="destination path of the overlap region of the chunk,\
# #               prefixes supported: file://, gs://, s3://.",
# #               required=True)
# # @click.option('--model_path', type=str, help="the path of convnet model", required=True)
# # @click.option('--net_path', type=str, help="the path of convnet weights", required=True)
# # @click.option('--gpu_ids', type=int, help="ids of cpus to use",
# #               cls=PythonLiteralOption, default=None)
# # @click.option('--framework', type=str, help="backend of deep learning framework, such as pytorch and pznet.",
# #               default='pytorch')
# # @click.option('--output_channels', type=int, help="number of convnet output channels", default=3)


# class InferenceEngine(object):
#     def __init__(self):
#         self.input_patch_size
#         pass
#     def __call__(input_patch):
#         pass

# class IdentityEngine(InferenceEngine):
#     def __init__(self, patch_size, model_path, net_path, patch_size):
#         # init stuff here
#         self.patch_size = patch_size

#     def __call__(input_patch):
#         return input_path * np.ones(patch_size)


# from collections import deque

# def get_face_neighbors(index):
#     pass

# def get_all_neighbors(index):
#     pass

# def blah(cloudvolume_local_core, cloudvolume_local_shell):
#     deque = deque()
#     dequeue.append((0,0,0))
#     while len(deque):
#         patch_index = deque.popleft()
#         patch_slice = get_slices(block_index)
#         core_slice = get_core_slice(patch_slice, overlap)
#         shell_slices = get_core_slice(patch_slice, overlap)
#         cloud_volume_local_core[core_slice] = patch_output[core_slice]
#         for shell_slice in shell_slices:
#             cloud_volume_local_shell[shell_slice] = patch_output[shell_slice]

#         for face_neighbor in get_face_neighbors(patch_index):
#             deque.append(face_neighbor)

#         deque.append(


# class BlockScheduler(object):
#     def __init__(self, input_volume, output_core_volume, output_shell_volume, inference_engine, patch_engine):
#         self.input_volume = input_volume
#         self.output_core_volume = output_core_volume
#         self.output_shell_volume = output_shell_volume
#         self.inference_engine = input_engine
#         self.patch_engine = patch_engine

#     def validate_slices(self, slices):
#         patch_size = self.inference_engine.patch_size
#         raise ValueError("not multiple of patch and overlap")


#     def execute(output_slice):
#         self.validate_slices(output_slice)


# def inference():
#     """
#     Run inference on a block
#     """
#     input_volume = CloudVolume(input_image_source)
#     output_core_volume = CloudVolume(output_core_destination)
#     output_shell_volume = CloudVolume(output_core_destination)
#     inference_engine = InferenceFactory.get(framework, model_path, net_path, patch_size)
#     patch_engine = PatchMasks.get(patch_type, patch_params) # TODO patch params

#     block_engine = BlockScheduler(input_volume, output_core_volume, output_shell_volume, inference_engine, patch_engine)


#     block_engine.execute(overlap, output_block_slice) # check to make sure matches with ng chunks and multiple of patch






#     print('inference')


# # @click.option('--input_overlap_sources', type=str, help="input source path(s) for overlap regions to blend \
# #               prefixes supported: file://, gs://, s3://. i.e. [\"gs://mybucket/left\", \"gs://mybucket/right\"]",
# #               required=True)
# # @click.option('--output_final_destination', type=str, help="final destination path of chunkflow blended inference,\
# #               prefixes supported: file://, gs://, s3://.",
# #               required=True)
# @main.command()
# def blend():
#     """
#     Blend chunk using overlap regions
#     """
# if __name__ == '__main__':
#     main()
