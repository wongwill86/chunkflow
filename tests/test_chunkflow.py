import numpy as np
from click.testing import CliRunner

from chunkflow.cli import main
from chunkflow.iterators import UnitIterator


def test_inference(cloudvolume_datasource_manager):
    runner = CliRunner()
    offset = cloudvolume_datasource_manager.input_datasource.voxel_offset[::-1]
    volume_shape = cloudvolume_datasource_manager.input_datasource.volume_size[::-1]
    task_shape = [3, 3, 3]

    dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offset, volume_shape))

    cloudvolume_datasource_manager.input_datasource[dataset_bounds] = np.ones(
        volume_shape, dtype=np.dtype(cloudvolume_datasource_manager.input_datasource.data_type))

    result = runner.invoke(main, [
        '--output_destination', cloudvolume_datasource_manager.output_datasource_core.layer_cloudpath,
        # '--output_overlap_destination', cloudvolume_datasource_manager.output_datasource_overlap.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', '[1, 1, 1]',
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        'inference',
        '--patch_shape', '[3, 3, 3]',
        '--framework', 'identity',
    ])

    np.set_printoptions(threshold=np.inf)

    data = (
        cloudvolume_datasource_manager.output_datasource_core[(slice(0, 3),) + dataset_bounds] +
        cloudvolume_datasource_manager.output_datasource_overlap[(slice(0, 3),) + dataset_bounds]
    )
    print(result.output)
    assert result.exit_code == 0
    assert result.exception is None
    assert np.prod(task_shape) * 3 == data.sum()


def test_blend(cloudvolume_datasource_manager):
    runner = CliRunner()
    offset = cloudvolume_datasource_manager.input_datasource.voxel_offset[::-1]
    volume_shape = cloudvolume_datasource_manager.input_datasource.volume_size[::-1]
    task_shape = (3, 3, 3)
    output_shape = (3,) + task_shape

    task_bounds = tuple(slice(o, o + s) for o, s in zip(offset, task_shape))
    dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offset, volume_shape))

    datasource = cloudvolume_datasource_manager.repository.get_datasource(task_shape)
    datasource[task_bounds] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))
    for neighbor in UnitIterator().get_all_neighbors(task_shape):
        datasource = cloudvolume_datasource_manager.repository.get_datasource(neighbor)
        datasource[task_bounds] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

    result = runner.invoke(main, [
        '--output_destination', cloudvolume_datasource_manager.output_datasource_core.layer_cloudpath,
        'task',
        '--task_offset_coordinates', list(offset),
        '--task_shape', list(task_shape),
        '--overlap', '[1, 1, 1]',
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        # '--output_overlap_destination', cloudvolume_datasource_manager.output_datasource_overlap.layer_cloudpath,
        'blend',
    ])

    np.set_printoptions(threshold=np.inf)

    print(result.output)
    assert result.exit_code == 0
    assert result.exception is None
    assert 3 ** len(task_shape) * 7 * 3 == \
        cloudvolume_datasource_manager.output_datasource_core[dataset_bounds].sum()


def test_check(cloudvolume_datasource_manager):
    runner = CliRunner()
    result = runner.invoke(main, [
        '--output_destination', cloudvolume_datasource_manager.output_datasource_core.layer_cloudpath,
        'cloudvolume',
        '--patch_shape', [1024, 1024, 256],
        '--overlap', [32, 32, 8],
        '--output_channels', '3',
        'check',
    ])
    print(result.output)
    assert result.exit_code == -1
    # TODO add more use case
