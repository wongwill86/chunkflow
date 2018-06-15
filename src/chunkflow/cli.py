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
from chunkflow.cloudvolume_datasource import (
    CloudVolumeCZYX,
    CloudVolumeDatasourceRepository,
    default_intermediate_name,
    default_overlap_datasource,
    default_overlap_name
)
from chunkflow.cloudvolume_helpers import get_possible_chunk_sizes, valid_cloudvolume
from chunkflow.datasource_manager import DatasourceManager, get_mod_index
from chunkflow.iterators import UnitIterator
from chunkflow.models import Block
from chunkflow.streams import create_blend_stream, create_inference_and_blend_stream


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
@click.option('--output_destination', type=str, help="destination path for the valid output of the chunk,\
              prefixes supported: file://, gs://, s3://.",
              required=True)
@click.pass_context
def main(ctx, **kwargs):
    obj = {}
    obj.update(kwargs)
    ctx.obj = obj
    pass


@main.group()
@click.option('--task_offset_coordinates', type=list, help="the start coordinates to run task (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--task_shape', type=list, help="shape of the input task to run (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--overlap', type=list,
              help="overlap across this task with other tasks, assumed same as patch overlap (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--input_image_source', type=str, help="input image source path, i.e. file://, gs://, or s3://.",
              required=True)
@click.pass_obj
def task(obj, **kwargs):
    """
    Task related options
    """
    print('Setting up datasource manager')
    obj.update(kwargs)

    obj['task_bounds'] = tuple(slice(o, o + sh) for o, sh in zip(
        kwargs['task_offset_coordinates'],
        kwargs['task_shape']
    ))

    input_cloudvolume = CloudVolumeCZYX(
        kwargs['input_image_source'], cache=False, non_aligned_writes=True, fill_missing=True)

    output_cloudvolume = CloudVolumeCZYX(
        obj['output_destination'], cache=False, non_aligned_writes=True, fill_missing=True)

    output_cloudvolume_overlap = default_overlap_datasource(output_cloudvolume)

    repository = CloudVolumeDatasourceRepository(input_cloudvolume, output_cloudvolume, output_cloudvolume_overlap)

    datasource_manager = DatasourceManager(repository)
    obj['datasource_manager'] = datasource_manager


@task.command()
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
    output_datasource = obj['datasource_manager'].output_datasource
    block = Block(bounds=obj['task_bounds'], chunk_shape=patch_shape, overlap=obj['overlap'])
    task_stream = create_inference_and_blend_stream(
        block=block,
        inference_operation=IdentityInference(
            output_channels=output_datasource.num_channels,
            output_data_type=output_datasource.data_type
        ),
        blend_operation=AverageBlend(block),
        datasource_manager=obj['datasource_manager']
    )

    BlockProcessor().process(block, task_stream)
    print('Finished inference!')


@task.command()
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

    print('Finished blend!')


@main.group()
@click.option('--patch_shape', type=list, help="convnet input patch shape",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--overlap', type=list,
              help="overlap across this task with other tasks, assumed same as patch overlap (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--output_channels', type=int, help="number of convnet output channels", default=3)
@click.option('--intermediates/--no-intermediates', help="Option to consider intermediate datasources", default=False)
@click.pass_obj
def cloudvolume(obj, **kwargs):
    """
    Cloudvolume related options
    """
    obj.update(kwargs)

    overlap = kwargs['overlap']
    patch_shape = kwargs['patch_shape']
    chunk_shape_options = get_possible_chunk_sizes(overlap, patch_shape)
    obj['chunk_shape_options'] = chunk_shape_options

    print('Finished Prepare')


@cloudvolume.command()
@click.pass_obj
def check(obj):
    """
    Check if output destination exists and if the chunk sizes are correct (use --intermediates to check intermediates)
    """
    print('Checking cloudvolume ...')
    output_destination = obj['output_destination']
    assert valid_cloudvolume(output_destination, obj['chunk_shape_options'])
    assert valid_cloudvolume(default_overlap_name(output_destination), obj['chunk_shape_options'])

    if obj['intermediates']:
        output_cloudvolume = CloudVolumeCZYX(obj['output_destination'])
        for neighbor in UnitIterator().get_all_neighbors((0,) * len(obj['patch_shape'])):
            assert valid_cloudvolume(default_intermediate_name(output_destination, get_mod_index(neighbor)),
                                     obj['chunk_shape_options'])
    print('Done cloudvolume!')


@cloudvolume.command()
@click.pass_obj
def create(obj):
    """
    Try to create output destinations(use --intermediates to create intermediates)
    """
    print('Creating cloudvolume ...')
    datasource_names = []
    datasource_names.append(obj['output_destination'])
    datasource_names.append(default_overlap_name(obj['output_destination']))

    if obj['intermediates']:
        datasource_names.extend([
            default_intermediate_name(obj['output_destination'], neighbor)
            for neighbor in UnitIterator().get_all_neighbors((0,) * len(obj['patch_shape']))
        ])

    for datasource_name in datasource_names:
        print(datasource_name)


    # try:
    #     CloudVolumeCZYX(output_destination)
    #     if not any(tuple(actual_chunk_size) == chunk_shape for chunk_shape in chunk_shape_options):
    #         print('Warning: %s already has incorrect chunk size %s. Please reformat with one of these chunk sizes: %s' %
    #               (cloudvolume.layer_cloudpath, actual_chunk_size, chunk_shape_options))
    # except ValueError:



    # assert valid_cloudvolume(default_overlap_name(output_destination), obj['chunk_shape_options'])

    print('Done creating cloudvolume!')


if __name__ == '__main__':
    main()
