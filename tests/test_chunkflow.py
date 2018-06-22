import numpy as np
from click.testing import CliRunner

from chunkflow.cli import main
from chunkflow.cloudvolume_datasource import (
    CloudVolumeCZYX,
    default_intermediate_datasource,
    default_overlap_datasource
)
from chunkflow.datasource_manager import get_all_mod_index


def test_inference(cloudvolume_datasource_manager):
    runner = CliRunner()
    offset = cloudvolume_datasource_manager.input_datasource.voxel_offset[::-1]
    volume_shape = cloudvolume_datasource_manager.input_datasource.volume_size[::-1]
    task_shape = [3, 3, 3]

    dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offset, volume_shape))

    cloudvolume_datasource_manager.input_datasource[dataset_bounds] = np.ones(
        volume_shape, dtype=np.dtype(cloudvolume_datasource_manager.input_datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', cloudvolume_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', [1, 1, 1],
        '--intermediate_protocol', 'file://',
        'inference',
        '--patch_shape', [3, 3, 3],
        '--inference_framework', 'identity',
        '--blend_framework', 'average',
    ])

    np.set_printoptions(threshold=np.inf)

    data = (
        cloudvolume_datasource_manager.output_datasource[(slice(0, 3),) + dataset_bounds] +
        cloudvolume_datasource_manager.output_datasource_overlap[(slice(0, 3),) + dataset_bounds]
    )
    print(result.output)
    assert result.exit_code == 0
    assert result.exception is None
    assert np.prod(task_shape) * 3 == data.sum()


def test_blend_with_offset_top_edge_task(cloudvolume_datasource_manager):
    runner = CliRunner()
    offset = cloudvolume_datasource_manager.input_datasource.voxel_offset[::-1]
    volume_shape = cloudvolume_datasource_manager.input_datasource.volume_size[::-1]
    task_shape = (3, 3, 3)
    output_shape = (3,) + task_shape

    task_bounds = tuple(slice(o, o + s) for o, s in zip(offset, task_shape))
    dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offset, volume_shape))

    cloudvolume_datasource_manager.repository.create_intermediate_datasources(task_shape)
    for datasource in cloudvolume_datasource_manager.repository.intermediate_datasources.values():
        datasource[task_bounds] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', cloudvolume_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', [1, 1, 1],
        'blend',
        '--voxel_offset', list(offset),
        '--volume_size', [3, 3, 3],
    ])

    print(result.output)
    print(cloudvolume_datasource_manager.output_datasource[dataset_bounds])
    assert result.exit_code == 0
    assert result.exception is None
    # Includes top left edge task
    assert (3 ** len(task_shape)) * (3 ** len(task_shape) - 1) * 3 == \
        cloudvolume_datasource_manager.output_datasource[dataset_bounds].sum()


def test_blend_with_offset_non_top_edge_task(cloudvolume_datasource_manager):
    runner = CliRunner()
    offset = cloudvolume_datasource_manager.input_datasource.voxel_offset[::-1]
    volume_shape = cloudvolume_datasource_manager.input_datasource.volume_size[::-1]
    task_shape = (3, 3, 3)
    output_shape = (3,) + task_shape

    task_bounds = tuple(slice(o, o + s) for o, s in zip(offset, task_shape))
    dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offset, volume_shape))

    cloudvolume_datasource_manager.repository.create_intermediate_datasources(task_shape)
    for datasource in cloudvolume_datasource_manager.repository.intermediate_datasources.values():
        datasource[task_bounds] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', cloudvolume_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', [1, 1, 1],
        'blend',
        '--voxel_offset', list(o - s + 1 for o, s in zip(offset, task_shape)),
        '--volume_size', [7, 7, 7],
    ])

    print(result.output)
    print(cloudvolume_datasource_manager.output_datasource[dataset_bounds])
    assert result.exit_code == 0
    assert result.exception is None
    assert 3 ** len(task_shape) * 7 * 3 == \
        cloudvolume_datasource_manager.output_datasource[dataset_bounds].sum()


def test_blend_no_offset(cloudvolume_datasource_manager):
    runner = CliRunner()
    offset = cloudvolume_datasource_manager.input_datasource.voxel_offset[::-1]
    volume_shape = cloudvolume_datasource_manager.input_datasource.volume_size[::-1]
    task_shape = (3, 3, 3)
    output_shape = (3,) + task_shape

    task_bounds = tuple(slice(o, o + s) for o, s in zip(offset, task_shape))
    dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offset, volume_shape))

    cloudvolume_datasource_manager.repository.create_intermediate_datasources(task_shape)
    for datasource in cloudvolume_datasource_manager.repository.intermediate_datasources.values():
        datasource[task_bounds] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', cloudvolume_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', [1, 1, 1],
        'blend',
    ])

    np.set_printoptions(threshold=np.inf)

    print(result.output)
    print(cloudvolume_datasource_manager.output_datasource[dataset_bounds])
    assert result.exit_code == 0
    assert result.exception is None
    assert 3 ** len(task_shape) * 7 * 3 == \
        cloudvolume_datasource_manager.output_datasource[dataset_bounds].sum()


def test_blend_bad_param(cloudvolume_datasource_manager):
    runner = CliRunner()
    offset = cloudvolume_datasource_manager.input_datasource.voxel_offset[::-1]
    task_shape = (3, 3, 3)
    output_shape = (3,) + task_shape

    task_bounds = tuple(slice(o, o + s) for o, s in zip(offset, task_shape))

    cloudvolume_datasource_manager.repository.create_intermediate_datasources(task_shape)
    for datasource in cloudvolume_datasource_manager.repository.intermediate_datasources.values():
        datasource[task_bounds] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

    result = runner.invoke(main, [
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', cloudvolume_datasource_manager.output_datasource.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', [1, 1, 1],
        'blend',
        '--voxel_offset', [1, 1, 1],
    ])

    np.set_printoptions(threshold=np.inf)

    print(result.output)
    assert result.exit_code == -1


def test_check(cloudvolume_datasource_manager, output_cloudvolume_intermediate):
    runner = CliRunner()
    result = runner.invoke(main, [
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', cloudvolume_datasource_manager.output_datasource.layer_cloudpath,
        'cloudvolume',
        '--patch_shape', [3, 3, 3],
        '--overlap', [1, 1, 1],
        '--num_channels', 3,
        '--intermediates',
        'check',
    ])
    print(result.output)
    assert result.exit_code == 0


def test_check_bad_chunksize(cloudvolume_datasource_manager, output_cloudvolume_intermediate):
    datasource = cloudvolume_datasource_manager.output_datasource
    info = datasource.info
    info['scales'][datasource.mip]['chunk_sizes'][0] = [3, 3, 3]
    datasource.commit_info()

    runner = CliRunner()
    result = runner.invoke(main, [
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', cloudvolume_datasource_manager.output_datasource.layer_cloudpath,
        'cloudvolume',
        '--patch_shape', [3, 3, 3],
        '--overlap', [1, 1, 1],
        '--num_channels', 3,
        '--intermediates',
        'check',
    ])
    print(result.output)
    assert result.exit_code == -1


def test_check_missing_intermediates_not_needed(cloudvolume_datasource_manager):
    runner = CliRunner()
    result = runner.invoke(main, [
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', cloudvolume_datasource_manager.output_datasource.layer_cloudpath,
        'cloudvolume',
        '--patch_shape', [3, 3, 3],
        '--overlap', [1, 1, 1],
        '--num_channels', 3,
        'check',
    ])
    print(result.output)
    assert result.exit_code == 0


def test_check_missing_intermediates_needed(cloudvolume_datasource_manager):
    runner = CliRunner()
    result = runner.invoke(main, [
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        '--output_destination', cloudvolume_datasource_manager.output_datasource.layer_cloudpath,
        'cloudvolume',
        '--patch_shape', [3, 3, 3],
        '--overlap', [1, 1, 1],
        '--num_channels', 3,
        '--intermediates',
        'check',
    ])
    print(result.output)
    assert result.exit_code == -1


def test_check_missing_cloudvolume():
    runner = CliRunner()
    result = runner.invoke(main, [
        '--input_image_source', 'badlayer',
        '--output_destination', 'badlayer',
        'cloudvolume',
        '--patch_shape', [3, 3, 3],
        '--overlap', [1, 1, 1],
        '--num_channels', 3,
        '--intermediates',
        'check',
    ])
    print(result.output)
    assert result.exit_code == -1


def test_create_cloudvolume(input_cloudvolume):
    output_destination = input_cloudvolume.layer_cloudpath + 'output/'
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '--input_image_source', input_cloudvolume.layer_cloudpath,
            '--output_destination', output_destination,
            'cloudvolume',
            '--patch_shape', [3, 3, 3],
            '--overlap', [1, 1, 1],
            '--num_channels', 3,
            '--intermediates',
            'create'
        ],
        input='yes\n1'
    )

    print(result.output)

    voxel_offset = input_cloudvolume.voxel_offset
    volume_size = input_cloudvolume.volume_size
    resolution = input_cloudvolume.resolution
    chunk_size = [1, 1, 1]  # runner input chooses option 1 for [1, 1, 1]
    num_channels = 3
    data_type = 'float32'

    output_cloudvolume = CloudVolumeCZYX(output_destination)
    output_overlap_cloudvolume = default_overlap_datasource(output_cloudvolume)

    cloudvolumes = [output_cloudvolume, output_overlap_cloudvolume]

    for mod_index in get_all_mod_index((0,) * len(chunk_size)):
        cloudvolumes.append(default_intermediate_datasource(output_cloudvolume, mod_index))

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
            '--patch_shape', [3, 3, 3],
            '--overlap', [1, 1, 1],
            '--num_channels', 3,
            '--intermediates',
            'create'
        ],
        # no input because we use the chunk size from existing cloudvolumes
    )

    print(result.output)

    voxel_offset = input_cloudvolume.voxel_offset
    volume_size = input_cloudvolume.volume_size
    resolution = input_cloudvolume.resolution
    chunk_size = output_cloudvolume.underlying
    num_channels = 3
    data_type = 'float32'

    output_overlap_cloudvolume = default_overlap_datasource(output_cloudvolume)

    cloudvolumes = [output_cloudvolume, output_overlap_cloudvolume]

    for mod_index in get_all_mod_index((0,) * len(chunk_size)):
        cloudvolumes.append(default_intermediate_datasource(output_cloudvolume, mod_index))

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
            '--patch_shape', [3, 3, 3],
            '--overlap', [1, 1, 1],
            '--num_channels', 3,
            '--intermediates',
            'create'
        ],
        input='yes\n1'
    )

    print(result.output)

    assert result.exit_code == -1
