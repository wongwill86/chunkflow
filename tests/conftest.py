import pytest
from cloudvolume import CloudVolume

from chunkflow.cloudvolume_datasource import CloudVolumeCZYX, CloudVolumeDatasourceRepository, default_overlap_name
from chunkflow.datasource_manager import DatasourceManager

TEMPLATE_INFO_ARGS = {
    'layer_type': 'image',
    'encoding': 'raw',
    'resolution': [1, 1, 1],
}


@pytest.fixture(scope='function')
def cloudvolume_factory(tmpdir):
    """
    Fixture for a cloudvolume factory that creates and instantiate a cloudvolume instance.
    All dimensions should be given in CZYX C-order and if cloud_volume class is given to be simply cloudvolume,
    chunk_size, volume_size, and voxel_offsets are reversed (convert to Fortran-order).
    """
    class CloudVolumeFactory:
        def __init__(self, tmpdir):
            self.tmpdir = tmpdir

        def create(self,
                   name,
                   data_type='uint16',
                   chunk_size=None,
                   volume_size=None,
                   voxel_offset=None,
                   num_channels=1,
                   cloudvolume_class=CloudVolumeCZYX):
            if chunk_size is None:
                chunk_size = [4, 8, 8]

            if volume_size is None:
                volume_size = [12, 24, 24]

            if voxel_offset is None:
                voxel_offset = [200, 100, 50]

            if cloudvolume_class == CloudVolumeCZYX:
                chunk_size = chunk_size[::-1]
                volume_size = volume_size[::-1]
                voxel_offset = voxel_offset[::-1]

            info_args = TEMPLATE_INFO_ARGS.copy()
            info_args['data_type'] = data_type
            info_args['chunk_size'] = chunk_size
            info_args['volume_size'] = volume_size
            info_args['voxel_offset'] = voxel_offset
            info_args['num_channels'] = num_channels
            info = CloudVolume.create_new_info(**info_args)

            directory = 'file://' + str(self.tmpdir) + name
            input_cloudvolume = cloudvolume_class(directory, info=info, cache=False, non_aligned_writes=True,
                                                  fill_missing=True, compress=False)
            input_cloudvolume.commit_info()
            return input_cloudvolume

    return CloudVolumeFactory(tmpdir)


@pytest.fixture(scope='function')
def cloudvolume_datasource_manager(cloudvolume_factory):
    volume_size = (7, 7, 7)
    voxel_offset = (200, 100, 50)
    cloud_volume_chunk_size = (2, 2, 2)
    input_data_type = 'uint8'
    output_data_type = 'float32'
    num_channels = 3

    input_cloudvolume = cloudvolume_factory.create(
        'input', data_type=input_data_type, volume_size=volume_size, chunk_size=cloud_volume_chunk_size,
        voxel_offset=voxel_offset)
    output_cloudvolume_core = cloudvolume_factory.create(
        'output', data_type=output_data_type, volume_size=volume_size, chunk_size=cloud_volume_chunk_size,
        num_channels=num_channels, voxel_offset=voxel_offset)
    output_cloudvolume_overlap = cloudvolume_factory.create(
        default_overlap_name('output'), data_type=output_data_type, volume_size=volume_size,
        chunk_size=cloud_volume_chunk_size, num_channels=num_channels)

    repository = CloudVolumeDatasourceRepository(input_cloudvolume, output_cloudvolume_core, output_cloudvolume_overlap)

    return DatasourceManager(repository)
