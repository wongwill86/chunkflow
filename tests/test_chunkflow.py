from click.testing import CliRunner

from chunkflow.cli import main


def test_main(cloudvolume_datasource_manager):
    runner = CliRunner()
    result = runner.invoke(main, [
        '--task_offset_coordinates', '[0, 0]',
        '--task_shape', '[7, 7]',
        '--overlap', '[1, 1]',
        '--output_channels', '1',
        '--input_image_source', cloudvolume_datasource_manager.input_datasource.layer_cloudpath,
        '--output_core_destination', cloudvolume_datasource_manager.output_datasource_core.layer_cloudpath,
        '--output_overlap_destination', cloudvolume_datasource_manager.output_datasource_overlap.layer_cloudpath,
        'blend'])

    print(result.output)
    print(result.exception)
    assert result.output == '()\n'

    assert result.exit_code == 0
# def test_blah():
#     runner = CliRunner()
#     result = runner.invoke(main, ['--task_offset_coordinates', '(2,)', 'blend'])
#     print(result.output)

#     assert result.output == '()\n'
#     assert result.exit_code == 0
