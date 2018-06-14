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
from rx import Observable

from chunkflow.block_processor import BlockProcessor
from chunkflow.chunk_operations.blend_operation import AverageBlend
from chunkflow.chunk_operations.inference_operation import IdentityInference
from chunkflow.cloudvolume_datasource import CloudVolumeCZYX
from chunkflow.cloudvolume_datasource import CloudVolumeDatasourceRepository
from chunkflow.datasource_manager import DatasourceManager
from chunkflow.models import Block
from chunkflow.streams import create_blend_stream
from chunkflow.streams import create_inference_and_blend_stream


# https://stackoverflow.com/a/47730333
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            if type(value) == str:
                return ast.literal_eval(value)
            else:
                return value
        except Exception:
            raise click.BadParameter(value)


def validate_literal(ctx, param, value):
    if not isinstance(value, param.type.func):
        raise click.BadParameter(value)
    return value


@click.group()
@click.option('--task_offset_coordinates', type=list, help="the start coordinates to run task (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--task_shape', type=list, help="shape of the input task to run on (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--overlap', type=list, help="overlap across this task with other tasks (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--output_channels', type=int, help="number of convnet output channels", default=3)
@click.option('--input_image_source', type=str, help="input image source path, i.e. file://, gs://, or s3://.",
              required=True)
@click.option('--output_core_destination', type=str, help="destination path for the valid core output of the chunk,\
              prefixes supported: file://, gs://, s3://.",
              required=True)
@click.option('--output_overlap_destination', type=str, help="destination path of the overlap region of the chunk,\
              prefixes supported: file://, gs://, s3://.",
              required=True)
@click.pass_context
def main(ctx, **kwargs):
    """
    Set up configuration
    """
    obj = {}

    print('Setting up datasource manager')

    obj['task_bounds'] = tuple(slice(o, o + sh) for o, sh in zip(
        kwargs['task_offset_coordinates'],
        kwargs['task_shape']
    ))

    input_cloudvolume = CloudVolumeCZYX(
        kwargs['input_image_source'], cache=False, non_aligned_writes=True, fill_missing=True)
    output_cloudvolume_core = CloudVolumeCZYX(
        kwargs['output_core_destination'], cache=False, non_aligned_writes=True, fill_missing=True)
    output_cloudvolume_overlap = CloudVolumeCZYX(
        kwargs['output_overlap_destination'], cache=False, non_aligned_writes=True, fill_missing=True)
    repository = CloudVolumeDatasourceRepository(input_cloudvolume, output_cloudvolume_core, output_cloudvolume_overlap)

    datasource_manager = DatasourceManager(repository)

    obj.update(kwargs)
    obj['datasource_manager'] = datasource_manager

    ctx.obj = obj


@main.command()
@click.option('--patch_shape', type=list, help="convnet input patch shape",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--framework', type=str, help="backend of deep learning framework, such as pytorch and pznet.",
              default='cpytorch')
@click.option('--model_path', type=str, help="the path of convnet model")
@click.option('--net_path', type=str, help="the path of convnet weights")
@click.option('--accelerator_ids', type=list, help="ids of cpus/gpus to use",
              cls=PythonLiteralOption, callback=validate_literal, default=[1])
@click.pass_obj
def inference(obj, patch_shape, framework, model_path, net_path, accelerator_ids):
    """
    Run inference on task
    """
    print('Running inference ...')
    block = Block(bounds=obj['task_bounds'], chunk_shape=patch_shape, overlap=obj['overlap'])
    task_stream = create_inference_and_blend_stream(
        block=block,
        inference_operation=IdentityInference(
            output_channels=obj['output_channels'],
            output_data_type=obj['datasource_manager'].output_datasource_core.data_type
        ),
        blend_operation=AverageBlend(block),
        datasource_manager=obj['datasource_manager']
    )

    BlockProcessor().process(block, task_stream)
    print('Finished inference ...')


@main.command()
@click.pass_obj
def blend(obj):
    """
    Blend chunk using overlap regions
    """
    print('Blending ...')

    datasource_manager = obj['datasource_manager']

    input_datasource = datasource_manager.repository.input_datasource
    offset_fortran = input_datasource.voxel_offset
    dataset_shape_fortran = input_datasource.volume_size

    offset_c = offset_fortran[::-1]
    dataset_shape_c = dataset_shape_fortran[::-1]

    dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offset_c, dataset_shape_c))
    block = Block(bounds=dataset_bounds, chunk_shape=obj['task_shape'], overlap=obj['overlap'])

    datasource_manager.repository.create_intermediate_datasources(obj['task_shape'])
    blend_stream = create_blend_stream(block, datasource_manager)

    chunk_index = block.slices_to_unit_index(obj['task_bounds'])
    chunk = block.unit_index_to_chunk(chunk_index)
    Observable.just(chunk).flat_map(blend_stream).subscribe(print)

    print('Finished blend ...')


if __name__ == '__main__':
    main()
