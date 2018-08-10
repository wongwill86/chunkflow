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
import multiprocessing

import click
from chunkblocks.models import Block
from rx import Observable
from rx.concurrency import ThreadPoolScheduler

from chunkflow.block_processor import BlockProcessor
from chunkflow.chunk_operations.blend_operation import BlendFactory
from chunkflow.chunk_operations.inference_operation import InferenceFactory
from chunkflow.cloudvolume_datasource import (
    CloudVolumeCZYX,
    CloudVolumeDatasourceRepository,
    create_buffered_cloudvolumeCZYX,
    default_overlap_name
)
from chunkflow.cloudvolume_helpers import create_cloudvolume, get_possible_chunk_sizes, valid_cloudvolume
from chunkflow.datasource_manager import DatasourceManager, get_absolute_index, get_all_mod_index
from chunkflow.sparse_matrix_datasource import SparseMatrixDatasourceRepository
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
    if not param.required and value is None:
        return value

    if not isinstance(value, param.type.func):
        raise click.BadParameter(value)

    return value


@click.group()
@click.option('--input_image_source', type=str, help="input image source path, i.e. file://, gs://, or s3://.",
              required=True)
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
@click.option('--overlap_protocol', type=str, help="cloudvolume protocol to use for overlap cloudvolumes",
              default=None)
@click.option('--threads', type=int, help="number of threads to use",
              default=multiprocessing.cpu_count())
@click.pass_obj
def task(obj, **kwargs):
    """
    Task related options
    """
    print('Setting up datasource manager')
    obj.update(kwargs)

    obj['task_bounds'] = tuple(slice(o, o + sh) for o, sh in zip(
        obj['task_offset_coordinates'],
        obj['task_shape']
    ))

    input_cloudvolume = CloudVolumeCZYX(
        obj['input_image_source'], cache=True, non_aligned_writes=True, fill_missing=True)
    output_cloudvolume_final = CloudVolumeCZYX(
        obj['output_destination'], cache=False, non_aligned_writes=True, fill_missing=True)
    block_repository = CloudVolumeDatasourceRepository(
        input_cloudvolume=input_cloudvolume, output_cloudvolume=output_cloudvolume_final)

    absolute_index = get_absolute_index(obj['task_offset_coordinates'], obj['overlap'], obj['task_shape'])
    output_cloudvolume_overlap = block_repository.get_datasource(absolute_index)

    chunk_repository = CloudVolumeDatasourceRepository(
        input_cloudvolume,
        output_cloudvolume=create_buffered_cloudvolumeCZYX(output_cloudvolume_overlap),
        output_cloudvolume_final=create_buffered_cloudvolumeCZYX(output_cloudvolume_final),
        overlap_protocol=obj['overlap_protocol']
    )

    obj['block_cloudvolume_repository'] = block_repository
    obj['chunk_cloudvolume_repository'] = chunk_repository

    threads = obj['threads']
    if threads > 1:
        scheduler = ThreadPoolScheduler(threads)
    else:
        scheduler = None
    obj['scheduler'] = scheduler


@task.command()
@click.option('--patch_shape', type=list, help="convnet input patch shape (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--inference_framework', type=str, help="backend of deep learning framework, such as pytorch and pznet.",
              default='cpytorch')
@click.option('--blend_framework', type=str, help="What blend method to use",
              default='average')
@click.option('--model_path', type=str, help="the path of convnet model")
@click.option('--net_path', type=str, help="the path of convnet weights")
@click.option('--accelerator_ids', type=list, help="ids of cpus/gpus to use",
              cls=PythonLiteralOption, callback=validate_literal, default=[1])
@click.pass_obj
def inference(obj, patch_shape, inference_framework, blend_framework, model_path, net_path, accelerator_ids):
    """
    Run inference on task
    """
    print('Running inference ...')
    block = Block(bounds=obj['task_bounds'], chunk_shape=patch_shape, overlap=obj['overlap'])

    chunk_repository = obj['chunk_cloudvolume_repository']
    datasource_manager = DatasourceManager(
        SparseMatrixDatasourceRepository(
            input_datasource=chunk_repository.input_datasource,
            output_datasource=chunk_repository.output_datasource,
            output_datasource_final=chunk_repository.output_datasource_final,
            num_channels=chunk_repository.output_datasource_final.num_channels,
            block=block
        )
    )

    print('Using output_datasource', datasource_manager.output_datasource.layer_cloudpath)
    print('Using output_datasource_final', datasource_manager.output_datasource_final.layer_cloudpath)

    output_datasource = datasource_manager.output_datasource
    inference_factory = InferenceFactory(patch_shape, output_channels=output_datasource.num_channels,
                                         output_data_type=output_datasource.data_type)
    blend_factory = BlendFactory(block)

    task_stream = create_inference_and_blend_stream(
        block=block,
        inference_operation=inference_factory.get_operation(inference_framework, model_path, net_path, accelerator_ids),
        blend_operation=blend_factory.get_operation(blend_framework),
        datasource_manager=datasource_manager,
        scheduler=obj['scheduler']
    )

    BlockProcessor(block).process(task_stream)
    print('Finished inference!')


@task.command()
@click.option('--volume_size', type=list, help="Total size of volume data (ZYX order). Used to determine if overlap "
              "region needs to be included at dataset boundaries, i.e. if current task is at the boundary of region of "
              "interest or completely inside. MUST be specified with --voxel_offset",
              cls=PythonLiteralOption, callback=validate_literal, default=None)
@click.option('--voxel_offset', type=list, help="Beginning offset coordinates of volume data (ZYX order). Used to "
              "determine if overlap region needs to be included at dataset_boundaries, i.e. if current task is at the "
              "boundary of the region of interest or completely inside. MUST be specified with --voxel_offset",
              cls=PythonLiteralOption, callback=validate_literal, default=None)
@click.pass_obj
def blend(obj, **kwargs):
    """
    Blend chunk using overlap regions
    """
    print('Blending ...')
    obj.update(kwargs)
    task_shape = obj['task_shape']

    if obj['volume_size'] is not None and obj['voxel_offset'] is not None:
        dataset_bounds = tuple(slice(o, o + s) for o, s in zip(obj['voxel_offset'], obj['volume_size']))
    elif obj['volume_size'] is not None or obj['voxel_offset'] is not None:
        raise ValueError("MUST specify both volume_size AND voxel_offset")
    else:
        # assume this task is completely inside so we blend all overlap regions
        task_offset = obj['task_offset_coordinates']
        task_shape = obj['task_shape']
        overlap = obj['overlap']
        dataset_bounds = tuple(slice(o - (s - olap), o + (s - olap) + s) for o, s, olap in zip(
            task_offset, task_shape, overlap))
        print(dataset_bounds)

    datasource_manager = DatasourceManager(obj['block_cloudvolume_repository'])

    print(obj['voxel_offset'])
    print(obj['volume_size'])
    print('dataset_bounds', dataset_bounds)
    block = Block(bounds=dataset_bounds, chunk_shape=obj['task_shape'], overlap=obj['overlap'])

    datasource_manager.repository.create_overlap_datasources(obj['task_shape'])
    blend_stream = create_blend_stream(block, datasource_manager)

    chunk_index = block.chunk_slices_to_unit_index(obj['task_bounds'])
    chunk = block.unit_index_to_chunk(chunk_index)
    Observable.just(chunk).flat_map(blend_stream).subscribe(print)

    print('Finished blend!')


@main.group()
@click.option('--patch_shape', type=list, help="convnet input patch shape (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--overlap', type=list,
              help="overlap across this task with other tasks, assumed same as patch overlap (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--num_channels', type=int, help="number of convnet output channels", default=3)
@click.option('--check_overlaps/--no-check_overlaps', help="Option to consider overlap datasources", default=False)
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

    print('Finished cloudvolume prepare')


@cloudvolume.command()
@click.pass_obj
def check(obj):
    """
    Check if output destination exists and if the chunk sizes are correct (use --check_overlaps to check overlaps)
    """
    print('Checking cloudvolume ...')
    input_datasource = CloudVolumeCZYX(obj['input_image_source'])
    output_destination = obj['output_destination']
    assert valid_cloudvolume(output_destination, obj['chunk_shape_options'], input_datasource)

    if obj['check_overlaps']:
        for mod_index in get_all_mod_index((0,) * len(obj['patch_shape'])):
            assert valid_cloudvolume(default_overlap_name(output_destination, mod_index),
                                     obj['chunk_shape_options'], input_datasource)
    print('Done cloudvolume!')


@cloudvolume.command()
@click.option('--layer_type', type=str, help="Cloudvolume \"layer_type\"", default='image')
@click.option('--data_type', type=str, help="Data type of the output", default='float32')
@click.option('--chunk_size', type=list, help="Underlying chunk size to use (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, default=None)
@click.option('--volume_size', type=list, help="Total size of volume data (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, default=None)
@click.option('--voxel_offset', type=list, help="Beginning offset coordinates of volume data (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, default=None)
@click.pass_obj
def create(obj, **kwargs):
    """
    Try to create output destinations(use --check_overlaps to create check_overlaps)
    """
    print('Creating cloudvolume ...')
    obj.update(kwargs)
    datasource_names = []
    datasource_names.append(obj['output_destination'])

    if obj['check_overlaps']:
        datasource_names.extend([
            default_overlap_name(obj['output_destination'], mod_index)
            for mod_index in get_all_mod_index((0,) * len(obj['patch_shape']))
        ])

    chunk_size = obj.pop('chunk_size')
    missing_datasource_names = []
    for datasource_name in datasource_names:
        print('Examining: ', datasource_name)
        try:
            cloudvolume = CloudVolumeCZYX(datasource_name)
            if chunk_size is None:
                # cloudvolume is xyz, chunkflow is zyx
                chunk_size = cloudvolume.underlying[::-1]
                pass
            print('Found existing chunk size of %s' % (chunk_size,))
            assert tuple(chunk_size) == tuple(cloudvolume.underlying[::-1])
        except ValueError:
            missing_datasource_names.append(datasource_name)

    chunk_shape_options = obj['chunk_shape_options']

    if chunk_size is not None:
        assert any(tuple(chunk_size) == tuple(chunk_shape) for chunk_shape in chunk_shape_options)
    else:
        chunk_size = prompt_for_chunk_size(chunk_shape_options)

    if len(missing_datasource_names) > 0:
        print('Will use chunk sizes %s to create datasources: %s' % (chunk_size, '\n'.join(missing_datasource_names)))

        input_datasource = CloudVolumeCZYX(obj['input_image_source'])
        for datasource_name in missing_datasource_names:
            create_cloudvolume(datasource_name, chunk_size, input_datasource, **obj)
    else:
        print('Datasources already created with suitable chunk sizes')

    print('Done creating cloudvolume!')


def prompt_for_chunk_size(chunk_shape_options):
    print('Starting Fresh! No existing cloudvolumes found. What chunksize do you want to use?')
    for idx, chunk_shape_option in enumerate(chunk_shape_options):
        print('[%s]: %s' % (idx, chunk_shape_option))
    index = -1
    while not (index >= 0 and index < len(chunk_shape_options)):
        index = click.prompt('Please enter a valid integer selection', type=int)

    return chunk_shape_options[index]


if __name__ == '__main__':
    main()
