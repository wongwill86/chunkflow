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
import ast

import click


# https://stackoverflow.com/a/47730333
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            raise click.BadParameter(value)


@click.group()
# @click.option('--output_chunk_start', type=int, help="the start coordinates of final output block",
#               cls=PythonLiteralOption, required=True)
# @click.option('--output_chunk_shape', type=int, help="the size of output block",
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
