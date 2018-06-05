import pytest

import numpy as np
from cloudvolume import CloudVolume
from chunkflow.cloudvolume_datasource import CloudVolumeCZYX
from chunkflow.cloudvolume_datasource import CloudVolumeDatasourceRepository
from chunkflow.global_offset_array import GlobalOffsetArray


class TestCloudVolumeCZYX:

    def test_get(self, cloudvolume_factory):
        data_shape_fortran = [8, 8, 4]
        cv_fortran = cloudvolume_factory.create('test', chunk_size=data_shape_fortran, volume_size=data_shape_fortran,
                                                cloudvolume_class=CloudVolume)
        offset_fortran = cv_fortran.info['scales'][0]['voxel_offset']

        slices_fortran = tuple(slice(0 + o, d + o) for d, o in zip(data_shape_fortran, offset_fortran))
        slices_c = slices_fortran[::-1]

        # F ordered xyzc
        data_fortran = np.arange(0, np.product(data_shape_fortran), dtype=np.uint16).reshape(
            data_shape_fortran, order='F')
        cv_fortran[slices_fortran] = data_fortran
        expected = cv_fortran[slices_fortran]

        cv_c = CloudVolumeCZYX(cv_fortran.layer_cloudpath, non_aligned_writes=True, fill_missing=True, compress=False)
        actual = cv_c[slices_c]

        for c in range(0, 1):
            for z in range(slices_fortran[2].start, slices_fortran[2].stop):
                for y in range(slices_fortran[1].start, slices_fortran[1].stop):
                    for x in range(slices_fortran[0].start, slices_fortran[0].stop):
                        expected_index = tuple(i - o for i, o in zip((x, y, z, c), offset_fortran)) + (0,)
                        assert actual[c][z][y][x] == expected[expected_index]

    def test_set(self, cloudvolume_factory):
        data_shape_fortran = [8, 8, 4]
        data_shape_c = data_shape_fortran[::-1]
        cv_c = cloudvolume_factory.create('test', chunk_size=data_shape_c, volume_size=data_shape_c,
                                          cloudvolume_class=CloudVolumeCZYX)
        offset_fortran = cv_c.info['scales'][0]['voxel_offset']

        slices_fortran = tuple(slice(o, d + o) for d, o in zip(data_shape_fortran, offset_fortran))
        slices_c = slices_fortran[::-1]

        # C ordered czyx
        data_c = np.arange(0, np.product(data_shape_c), dtype=np.uint16).reshape(data_shape_c, order='C')
        cv_c[slices_c] = data_c
        data_c = np.expand_dims(data_c, 0)

        cv_fortran = CloudVolume(cv_c.layer_cloudpath, non_aligned_writes=True, fill_missing=True, compress=False)
        actual = cv_fortran[slices_fortran]

        for c in range(0, 1):
            for z in range(slices_fortran[2].start, slices_fortran[2].stop):
                for y in range(slices_fortran[1].start, slices_fortran[1].stop):
                    for x in range(slices_fortran[0].start, slices_fortran[0].stop):
                        expected_index = tuple(i - o for i, o in zip((x, y, z), offset_fortran)) + (0,)
                        assert actual[expected_index] == data_c[expected_index[::-1]]

    def test_multi_dimensional_conversions(self, cloudvolume_factory):
        input_cloudvolume = cloudvolume_factory.create('input')
        output_cloudvolume = cloudvolume_factory.create('output', num_channels=3)

        offset = input_cloudvolume.info['scales'][0]['voxel_offset'][::-1]

        sizes = (8, 16, 16)
        slices = tuple(slice(o, s + o) for o, s in zip(offset, sizes))

        # Setup Data
        input_data = np.arange(0, 8*16*16, dtype='uint16').reshape((8, 16, 16))
        input_cloudvolume[slices] = input_data

        # Pull Data
        input_data = input_cloudvolume[slices]

        global_offset = input_data.global_offset
        one = input_data
        two = input_data * 10
        three = input_data * 100

        output_cloudvolume[slices] = GlobalOffsetArray(np.stack((one, two, three)).squeeze(),
                                                       global_offset=global_offset)

        assert np.array_equal(output_cloudvolume[(0,) + slices], input_data)
        assert np.array_equal(output_cloudvolume[(1,) + slices], input_data * 10)
        assert np.array_equal(output_cloudvolume[(2,) + slices], input_data * 100)


class TestCloudVolumeDatasource:
    def test_fail_not_cloudvolumeczyx(self, cloudvolume_factory):
        input_cloudvolume = cloudvolume_factory.create('input', cloudvolume_class=CloudVolume)
        output_cloudvolume_core = cloudvolume_factory.create('output_core', cloudvolume_class=CloudVolume)
        output_cloudvolume_overlap = cloudvolume_factory.create('output_path_overlap', cloudvolume_class=CloudVolume)

        with pytest.raises(ValueError):
            CloudVolumeDatasourceRepository(input_cloudvolume, output_cloudvolume_core, output_cloudvolume_overlap)

    def test_create_mod_index(self, cloudvolume_factory):
        input_cloudvolume = cloudvolume_factory.create('input')
        output_cloudvolume_core = cloudvolume_factory.create('output_core')
        output_cloudvolume_overlap = cloudvolume_factory.create('output_path_overlap')

        repository = CloudVolumeDatasourceRepository(input_cloudvolume, output_cloudvolume_core,
                                                     output_cloudvolume_overlap)

        datasource = repository.get_datasource((10, 11, 12))

        assert repository.get_datasource((1, 2, 0)) == datasource
