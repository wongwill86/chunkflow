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
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
from chunkblocks.models import Block
from rx import Observable

from chunkflow.block_processor import BlockProcessor
from chunkflow.chunk_operations.blend_operation import BlendFactory
from chunkflow.chunk_operations.inference_operation import InferenceFactory
from chunkflow.cloudvolume_datasource import (
    CloudVolumeCZYX,
    CloudVolumeDatasourceManager,
    create_buffered_cloudvolumeCZYX,
    default_overlap_name
)
from chunkflow.cloudvolume_helpers import create_cloudvolume, get_possible_chunk_sizes, valid_cloudvolume
from chunkflow.datasource_manager import SparseOverlapRepository, get_absolute_index, get_all_mod_index
from chunkflow.streams import create_blend_stream, create_inference_and_blend_stream, create_preload_datasource_stream


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
        obj['input_image_source'], cache=False, fill_missing=True)
    output_cloudvolume_final = CloudVolumeCZYX(
        obj['output_destination'], cache=False, fill_missing=True, non_aligned_writes=True)

    obj['block_datasource_manager'] = CloudVolumeDatasourceManager(
        input_cloudvolume=input_cloudvolume, output_cloudvolume=output_cloudvolume_final,
        load_executor=ProcessPoolExecutor(), flush_executor=ProcessPoolExecutor())


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
    block_datasource_manager = obj['block_datasource_manager']

    block = Block(bounds=obj['task_bounds'], chunk_shape=patch_shape, overlap=obj['overlap'])

    absolute_index = get_absolute_index(obj['task_offset_coordinates'], obj['overlap'], obj['task_shape'])
    output_cloudvolume_overlap = block_datasource_manager.overlap_repository.get_datasource(absolute_index)

    chunk_datasource_manager = CloudVolumeDatasourceManager(
        block_datasource_manager.input_datasource,
        output_cloudvolume=output_cloudvolume_overlap,
        output_cloudvolume_final=block_datasource_manager.output_datasource,
        overlap_repository=SparseOverlapRepository(
            block=block,
            channel_dimensions=(output_cloudvolume_overlap.num_channels,),
            dtype=output_cloudvolume_overlap.dtype,
        ),
        buffer_generator=create_buffered_cloudvolumeCZYX,
        load_executor=ProcessPoolExecutor(),
        flush_executor=ProcessPoolExecutor()
    )

    print('Using output_datasource', chunk_datasource_manager.output_datasource.layer_cloudpath)
    print('Using output_datasource_final', chunk_datasource_manager.output_datasource_final.layer_cloudpath)

    output_datasource = chunk_datasource_manager.output_datasource
    inference_factory = InferenceFactory(patch_shape, output_channels=output_datasource.num_channels,
                                         output_data_type=output_datasource.data_type)
    blend_factory = BlendFactory(block)

    task_stream = create_inference_and_blend_stream(
        block=block,
        inference_operation=inference_factory.get_operation(inference_framework, model_path, net_path, accelerator_ids),
        blend_operation=blend_factory.get_operation(blend_framework),
        datasource_manager=chunk_datasource_manager,
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

    datasource_manager = obj['block_datasource_manager']
    datasource_manager.buffer_generator = create_buffered_cloudvolumeCZYX

    print(obj['voxel_offset'])
    print(obj['volume_size'])
    print('dataset_bounds', dataset_bounds)
    block = Block(bounds=dataset_bounds, chunk_shape=obj['task_shape'], overlap=obj['overlap'])

    datasource_manager.create_overlap_datasources(obj['task_shape'])
    blend_stream = create_blend_stream(block, datasource_manager)

    chunk_index = block.chunk_slices_to_unit_index(obj['task_bounds'])
    chunk = block.unit_index_to_chunk(chunk_index)

    (
        Observable.just(chunk)
        .flat_map(create_preload_datasource_stream(block, datasource_manager,
                                                   datasource_manager.output_datasource))
        .flat_map(blend_stream)
        .reduce(lambda x, y: x)
        # flushing everything returns a list of futures
        .map(lambda _: datasource_manager.flush(None, datasource=datasource_manager.output_datasource))
        .flat_map(lambda futures: as_completed(futures))
        .subscribe(lambda a: print('Completed Blend of datasource_chunk:!', a.result().unit_index))
    )

    print('Finished blend!')


@main.group()
@click.option('--patch_shape', type=list, help="convnet input patch shape (ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, required=True)
@click.option('--num_patches', type=list, help="how large of a task desired(ZYX order)",
              cls=PythonLiteralOption, callback=validate_literal, default=[4, 4, 4])
@click.option('--min_mips', type=int, help="number of mip levels expected (sets minimum chunk size)",
              default=4)
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
    chunk_shape_options = get_possible_chunk_sizes(overlap, patch_shape, obj['min_mips'], obj['num_patches'])
    obj['chunk_shape_options'] = chunk_shape_options
    obj['input_datasource'] = CloudVolumeCZYX(obj['input_image_source'])

    print('Finished cloudvolume prepare')


@cloudvolume.command()
@click.pass_obj
def check(obj):
    """
    Check if output destination exists and if the chunk sizes are correct (use --check_overlaps to check overlaps)
    """
    print('Checking cloudvolume ...')
    output_destination = obj['output_destination']
    assert valid_cloudvolume(output_destination, obj['chunk_shape_options'], obj['input_datasource'])

    if obj['check_overlaps']:
        for mod_index in get_all_mod_index((0,) * len(obj['patch_shape'])):
            assert valid_cloudvolume(default_overlap_name(output_destination, mod_index),
                                     obj['chunk_shape_options'], obj['input_datasource'])
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
            assert valid_cloudvolume(cloudvolume, obj['chunk_shape_options'], obj['input_datasource']), (
                'Invalid cloudvolume configuration detected! See warnings above'
            )

        except ValueError:
            missing_datasource_names.append(datasource_name)

    chunk_shape_options = obj['chunk_shape_options']

    if chunk_size is None:
        chunk_size = prompt_for_chunk_size(chunk_shape_options)

    if len(missing_datasource_names) > 0:
        print('Will use chunk sizes %s to create datasources: %s' % (chunk_size, '\n'.join(missing_datasource_names)))

        for datasource_name in missing_datasource_names:
            create_cloudvolume(datasource_name, chunk_size, **obj)
    else:
        print('Datasources already created with suitable chunk sizes')

    print('Done creating cloudvolume!')


def prompt_for_chunk_size(chunk_shape_options):
    print('Starting Fresh! No existing cloudvolumes found. What chunksize do you want to use?')
    dimensions = ['Z', 'Y', 'X']
    selection = []
    for dim, chunk_shape_option in zip(dimensions, chunk_shape_options):
        for idx, shape_option in enumerate(chunk_shape_option):
            print('[%s]: %s' % (idx, shape_option))

        option_index = -1
        while not (option_index >= 0 and option_index < len(chunk_shape_option)):
            option_index = click.prompt('Please enter size for dimension %s' % dim, type=int)

        selection.append(chunk_shape_option[option_index])

    return selection


if __name__ == '__main__':
    main(standalone=False)
