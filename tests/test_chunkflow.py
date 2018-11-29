import traceback

import numpy as np
from click.testing import CliRunner

from chunkflow.cli import main
from chunkflow.cloudvolume_datasource import CloudVolumeCZYX, default_overlap_datasource
from chunkflow.datasource_manager import get_all_mod_index


def test_inference(block_datasource_manager):
    runner = CliRunner()
    offset = block_datasource_manager.input_datasource.voxel_offset[::-1]
    volume_shape = block_datasource_manager.input_datasource.volume_size[::-1]
    overlap = [1, 4, 4]
    patch_shape = [5, 10, 10]
    num_chunks = [2, 2, 2]
    task_shape = tuple((ps - olap) * num + olap for ps, olap, num in zip(patch_shape, overlap,
                                                                         num_chunks))

    dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offset, volume_shape))

    block_datasource_manager.input_datasource[dataset_bounds] = np.ones(
        volume_shape, dtype=np.dtype(block_datasource_manager.input_datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', block_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', block_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', overlap,
        'inference',
        '--patch_shape', patch_shape,
        '--inference_framework', 'identity',
        '--blend_framework', 'average',
    ])

    np.set_printoptions(threshold=np.inf)

    print(result.output)

    # force print error (until click 7.0 https://github.com/pallets/click/issues/371)
    if result.exception is not None:
        print(''.join(traceback.format_exception(etype=type(result.exception), value=result.exception,
                                                 tb=result.exception.__traceback__)))

    assert result.exit_code == 0
    assert result.exception is None
    inner_bounds = tuple(slice(o + olap, o + s - olap) for s, o, olap in zip(task_shape, offset, overlap))
    inner_shape = tuple(sh - o * 2 for sh, o in zip(task_shape, overlap))

    assert np.prod(inner_shape) * 3 == block_datasource_manager.output_datasource[inner_bounds].sum()


def test_inference_bad_config(block_datasource_manager):
    runner = CliRunner()
    offset = block_datasource_manager.input_datasource.voxel_offset[::-1]
    volume_shape = block_datasource_manager.input_datasource.volume_size[::-1]
    overlap = [1, 4, 4]
    patch_shape = [5, 10, 10]
    num_chunks = [3, 2, 2]
    task_shape = tuple((ps - olap) * num + olap for ps, olap, num in zip(patch_shape, overlap,
                                                                         num_chunks))

    dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offset, volume_shape))

    block_datasource_manager.input_datasource[dataset_bounds] = np.ones(
        volume_shape, dtype=np.dtype(block_datasource_manager.input_datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', block_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', block_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', overlap,
        'inference',
        '--patch_shape', patch_shape,
        '--inference_framework', 'identity',
        '--blend_framework', 'average',
    ])

    assert result.exit_code != 0


def test_blend_with_offset_outer_chunk(block_datasource_manager):
    """
    Normal higher index overlap edges along a chunk are normally processed by the following task chunk. However, along
    the dataset boundaries there are no such task and therefore these overlap edges along the dataset edge must be
    processed in the chunk.
    """
    runner = CliRunner()
    offset = block_datasource_manager.input_datasource.voxel_offset[::-1]
    task_shape = (3, 15, 15)
    output_shape = (3,) + task_shape
    overlap = [1, 5, 5]

    # Test for the bottom corner
    task_offset = list(o + oo for o, oo in zip(offset, (4, 20, 20)))
    task_bounds = tuple(slice(o, o + s) for o, s in zip(task_offset, task_shape))

    block_datasource_manager.create_overlap_datasources(task_shape)
    for datasource in block_datasource_manager.overlap_repository.datasources.values():
        datasource[task_bounds] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

    core_shape = tuple(s - 2 * olap for s, olap in zip(task_shape, overlap))
    core_slice = tuple(slice(o + olap, o + s - olap) for o, olap, s in zip(task_offset, overlap, task_shape))
    block_datasource_manager.output_datasource[core_slice] = np.ones((3,) + core_shape,
                                                                     dtype=np.dtype(datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', block_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', block_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(task_offset),  # list(offset),
        '--task_shape', list(task_shape),
        '--overlap', list(overlap),
        'blend',
        '--voxel_offset', list(offset),
        '--volume_size', [7, 35, 35],
    ])

    print(result.output)

    #  force print error (until click 7.0 https://github.com/pallets/click/issues/371)
    if result.exception is not None:
        print(''.join(traceback.format_exception(etype=type(result.exception), value=result.exception,
                                                 tb=result.exception.__traceback__)))

    assert result.exit_code == 0
    assert result.exception is None
    # Includes top left edge task

    np.set_printoptions(threshold=np.inf, linewidth=250)

    # for the bottom corner, only the core is not filled
    # ((entire task - core) * unique mod indices + core shape) * # channels
    assert (
        ((np.product(task_shape) - np.product(core_shape)) * 2 ** len(task_shape) + np.product(core_shape)) * 3) == \
        block_datasource_manager.output_datasource[task_bounds].sum()


def test_blend_with_offset_inner_chunk(block_datasource_manager):
    """
    Process task chunk as an inner chunk with no regard for dataset boundary effects
    """
    runner = CliRunner()
    offset = block_datasource_manager.input_datasource.voxel_offset[::-1]
    volume_shape = block_datasource_manager.input_datasource.volume_size[::-1]
    task_shape = (3, 30, 30)
    overlap = [1, 10, 10]
    output_shape = (3,) + task_shape

    task_bounds = tuple(slice(o, o + s) for o, s in zip(offset, task_shape))
    dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offset, volume_shape))

    block_datasource_manager.create_overlap_datasources(task_shape)
    for datasource in block_datasource_manager.overlap_repository.datasources.values():
        datasource[task_bounds] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', block_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', block_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', overlap,
        'blend',
        '--voxel_offset', list(o - s + olap for o, s, olap in zip(offset, task_shape, overlap)),
        '--volume_size', [7, 70, 70],
    ])

    print(result.output)
    #  force print error (until click 7.0 https://github.com/pallets/click/issues/371)
    if result.exception is not None:
        print(''.join(traceback.format_exception(etype=type(result.exception), value=result.exception,
                                                 tb=result.exception.__traceback__)))
    assert result.exit_code == 0
    assert result.exception is None

    # overlap area * overlaps * unique mod indices * # channels
    assert np.product(overlap) * 7 * 2 ** 3 * 3 == \
        block_datasource_manager.output_datasource[dataset_bounds].sum()


def test_blend_no_offset(block_datasource_manager):
    """
    With no voxel_offset specified, process task chunk as an inner chunk with no regard for dataset boundary effects
    """
    runner = CliRunner()
    offset = block_datasource_manager.input_datasource.voxel_offset[::-1]
    task_shape = (3, 30, 30)
    output_shape = (3,) + task_shape
    overlap = [1, 10, 10]

    task_bounds = tuple(slice(o, o + s) for o, s in zip(offset, task_shape))

    block_datasource_manager.create_overlap_datasources(task_shape)
    for datasource in block_datasource_manager.overlap_repository.datasources.values():
        datasource[task_bounds] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', block_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', block_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', overlap,
        'blend',
    ])

    np.set_printoptions(threshold=np.inf, linewidth=250)

    print(result.output)
    #  force print error (until click 7.0 https://github.com/pallets/click/issues/371)
    if result.exception is not None:
        print(''.join(traceback.format_exception(etype=type(result.exception), value=result.exception,
                                                 tb=result.exception.__traceback__)))
    assert result.exit_code == 0
    assert result.exception is None

    slices = tuple(slice(o, s + o) for s, o in zip(task_shape, offset))
    print(block_datasource_manager.output_datasource[slices][0])

    # overlap area * overlaps * unique mod indices * # channels
    assert np.product(overlap) * 7 * 2 ** 3 * 3 == \
        block_datasource_manager.output_datasource[slices].sum()


def test_blend_bad_param(block_datasource_manager):
    runner = CliRunner()
    offset = block_datasource_manager.input_datasource.voxel_offset[::-1]
    task_shape = (3, 30, 30)
    output_shape = (3,) + task_shape

    task_bounds = tuple(slice(o, o + s) for o, s in zip(offset, task_shape))

    block_datasource_manager.create_overlap_datasources(task_shape)
    for datasource in block_datasource_manager.overlap_repository.datasources.values():
        datasource[task_bounds] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', block_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', block_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', [1, 10, 10],
        'blend',
        '--voxel_offset', [1, 10, 10],
    ])

    np.set_printoptions(threshold=np.inf)

    print(result.output)
    assert result.exit_code != 0


def test_check(block_datasource_manager, output_cloudvolume_overlaps):
    runner = CliRunner()
    result = runner.invoke(main, [
        '--input_image_source', block_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', block_datasource_manager.output_datasource.layer_cloudpath,
        'cloudvolume',
        '--patch_shape', [5, 10, 10],
        '--overlap', [1, 4, 4],
        '--num_channels', 3,
        '--check_overlaps',
        '--num_patches', [4, 4, 4],
        '--min_mips', 0,
        'check',
    ])
    print(result.output)
    #  force print error (until click 7.0 https://github.com/pallets/click/issues/371)
    if result.exception is not None:
        print(''.join(traceback.format_exception(etype=type(result.exception), value=result.exception,
                                                 tb=result.exception.__traceback__)))
    assert result.exit_code == 0


def test_check_bad_chunksize(block_datasource_manager, output_cloudvolume_overlap):
    datasource = block_datasource_manager.output_datasource
    info = datasource.info
    info['scales'][datasource.mip]['chunk_sizes'][0] = [30, 30, 3]
    datasource.commit_info()

    runner = CliRunner()
    result = runner.invoke(main, [
        '--input_image_source', block_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', block_datasource_manager.output_datasource.layer_cloudpath,
        'cloudvolume',
        '--patch_shape', [3, 30, 30],
        '--overlap', [1, 10, 10],
        '--num_channels', 3,
        '--check_overlaps',
        '--num_patches', [4, 4, 4],
        '--min_mips', 1,
        'check',
    ])
    print(result.output)
    assert result.exit_code != 0


def test_check_missing_overlaps_not_needed(block_datasource_manager):
    runner = CliRunner()
    result = runner.invoke(main, [
        '--input_image_source', block_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', block_datasource_manager.output_datasource.layer_cloudpath,
        'cloudvolume',
        '--patch_shape', [5, 10, 10],
        '--overlap', [1, 4, 4],
        '--num_channels', 3,
        '--num_patches', [4, 4, 4],
        '--min_mips', 0,
        'check',
    ])
    print(result.output)
    #  force print error (until click 7.0 https://github.com/pallets/click/issues/371)
    if result.exception is not None:
        print(''.join(traceback.format_exception(etype=type(result.exception), value=result.exception,
                                                 tb=result.exception.__traceback__)))
    assert result.exit_code == 0


def test_check_missing_overlaps_needed(block_datasource_manager):
    runner = CliRunner()
    result = runner.invoke(main, [
        '--input_image_source', block_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', block_datasource_manager.output_datasource.layer_cloudpath,
        'cloudvolume',
        '--patch_shape', [3, 30, 30],
        '--overlap', [1, 10, 10],
        '--num_channels', 3,
        '--check_overlaps',
        '--num_patches', [4, 4, 4],
        '--min_mips', 1,
        'check',
    ])
    print(result.output)
    assert result.exit_code != 0


def test_check_missing_cloudvolume():
    runner = CliRunner()
    result = runner.invoke(main, [
        '--input_image_source', 'badlayer',
        '--output_destination', 'badlayer',
        'cloudvolume',
        '--patch_shape', [3, 30, 30],
        '--overlap', [1, 10, 10],
        '--num_channels', 3,
        '--check_overlaps',
        '--num_patches', [4, 4, 4],
        '--min_mips', 1,
        'check',
    ])
    print(result.output)
    assert result.exit_code != 0


def test_create_cloudvolume(input_cloudvolume):
    output_destination = input_cloudvolume.layer_cloudpath + 'output/'
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '--input_image_source', input_cloudvolume.layer_cloudpath,
            '--output_destination', output_destination,
            'cloudvolume',
            '--patch_shape', [3, 30, 30],
            '--overlap', [1, 10, 10],
            '--num_channels', 3,
            '--check_overlaps',
            '--num_patches', [4, 4, 4],
            '--min_mips', 0,
            'create'
        ],
        input='yes\n1\n1\n1'
    )

    print(result.output)
    #  force print error (until click 7.0 https://github.com/pallets/click/issues/371)
    if result.exception is not None:
        print(''.join(traceback.format_exception(etype=type(result.exception), value=result.exception,
                                                 tb=result.exception.__traceback__)))

    voxel_offset = input_cloudvolume.voxel_offset
    volume_size = input_cloudvolume.volume_size
    resolution = input_cloudvolume.resolution
    chunk_size = [40, 40, 4]  # runner input chooses option 1 for [1, 10, 10], then reverse becuase cv is xyzc
    num_channels = 3
    data_type = 'float32'

    output_cloudvolume = CloudVolumeCZYX(output_destination)

    cloudvolumes = [output_cloudvolume]

    for mod_index in get_all_mod_index((0,) * len(chunk_size)):
        cloudvolumes.append(default_overlap_datasource(output_cloudvolume, mod_index))

    for cloudvolume in cloudvolumes:
        assert tuple(voxel_offset) == tuple(cloudvolume.voxel_offset)
        assert tuple(volume_size) == tuple(cloudvolume.volume_size)
        assert tuple(resolution) == tuple(cloudvolume.resolution)
        assert tuple(chunk_size) == tuple(cloudvolume.underlying)
        assert num_channels == cloudvolume.num_channels
        assert data_type == cloudvolume.data_type

    assert result.exit_code == 0


def test_create_only_some_cloudvolume(input_cloudvolume, output_cloudvolume, output_cloudvolume_overlap):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '--input_image_source', input_cloudvolume.layer_cloudpath,
            '--output_destination', output_cloudvolume.layer_cloudpath,
            'cloudvolume',
            '--patch_shape', [5, 10, 10],
            '--overlap', [1, 4, 4],
            '--num_channels', 3,
            '--check_overlaps',
            '--num_patches', [4, 4, 4],
            '--min_mips', 0,
            'create'
        ],
        # no input because we use the chunk size from existing cloudvolumes
    )

    print(result.output)
    #  force print error (until click 7.0 https://github.com/pallets/click/issues/371)
    if result.exception is not None:
        print(''.join(traceback.format_exception(etype=type(result.exception), value=result.exception,
                                                 tb=result.exception.__traceback__)))

    voxel_offset = input_cloudvolume.voxel_offset
    volume_size = input_cloudvolume.volume_size
    resolution = input_cloudvolume.resolution
    chunk_size = output_cloudvolume.underlying
    num_channels = 3
    data_type = 'float32'

    output_overlap_cloudvolume = default_overlap_datasource(output_cloudvolume, (0,) * len(volume_size))

    cloudvolumes = [output_cloudvolume, output_overlap_cloudvolume]

    for mod_index in get_all_mod_index((0,) * len(chunk_size)):
        cloudvolumes.append(default_overlap_datasource(output_cloudvolume, mod_index))

    for cloudvolume in cloudvolumes:
        assert tuple(voxel_offset) == tuple(cloudvolume.voxel_offset)
        assert tuple(volume_size) == tuple(cloudvolume.volume_size)
        assert tuple(resolution) == tuple(cloudvolume.resolution)
        assert tuple(chunk_size) == tuple(cloudvolume.underlying)
        assert num_channels == cloudvolume.num_channels
        assert data_type == cloudvolume.data_type

    assert result.exit_code == 0


def test_create_cloudvolume_mixed_chunk_size(input_cloudvolume, output_cloudvolume, output_cloudvolume_overlap):
    # force output_cloudvolume to use a weird chunk size
    new_chunk_size = [3, 3, 3]
    output_cloudvolume.info['scales'][output_cloudvolume.mip]['chunk_sizes'][0] = new_chunk_size
    output_cloudvolume.commit_info()
    # just make sure the changes are propagated correctly
    assert tuple(new_chunk_size) == tuple(output_cloudvolume.underlying)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '--input_image_source', input_cloudvolume.layer_cloudpath,
            '--output_destination', output_cloudvolume.layer_cloudpath,
            'cloudvolume',
            '--patch_shape', [3, 30, 30],
            '--overlap', [1, 10, 10],
            '--num_channels', 3,
            '--check_overlaps',
            '--min_mips', 1,
            'create'
        ],
        input='yes\n1,1,1'
    )

    print(result.output)

    assert result.exit_code != 0
