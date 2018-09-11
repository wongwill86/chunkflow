import numpy as np
import pytest
from chunkblocks.global_offset_array import GlobalOffsetArray
from chunkblocks.models import Block
from cloudvolume import CloudVolume

from chunkflow.cloudvolume_datasource import (
    CloudVolumeCZYX,
    CloudVolumeDatasourceManager,
    create_buffered_cloudvolumeCZYX,
    default_overlap_name
)


class TestCloudVolumeCZYX:
    def test_get(self, cloudvolume_factory):
        data_shape_fortran = [8, 8, 4]
        cv_fortran = cloudvolume_factory.create('test', chunk_size=data_shape_fortran, volume_size=data_shape_fortran,
                                                cloudvolume_class=CloudVolume)
        offsets_fortran = cv_fortran.voxel_offset

        slices_fortran = tuple(slice(0 + o, d + o) for d, o in zip(data_shape_fortran, offsets_fortran))
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
                        expected_index = tuple(i - o for i, o in zip((x, y, z, c), offsets_fortran)) + (0,)
                        assert actual[c][z][y][x] == expected[expected_index]

    def test_set(self, cloudvolume_factory):
        data_shape_fortran = [8, 8, 4]
        data_shape_c = data_shape_fortran[::-1]
        cv_c = cloudvolume_factory.create('test', chunk_size=data_shape_c, volume_size=data_shape_c,
                                          cloudvolume_class=CloudVolumeCZYX)
        offsets_fortran = cv_c.voxel_offset

        slices_fortran = tuple(slice(o, d + o) for d, o in zip(data_shape_fortran, offsets_fortran))
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
                        expected_index = tuple(i - o for i, o in zip((x, y, z), offsets_fortran)) + (0,)
                        assert actual[expected_index] == data_c[expected_index[::-1]]

    def test_multi_dimensional_conversions(self, cloudvolume_factory):
        input_cloudvolume = cloudvolume_factory.create('input')
        output_cloudvolume = cloudvolume_factory.create('output', num_channels=3)

        offset = input_cloudvolume.voxel_offset[::-1]

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
    def test_create_with_final(self, cloudvolume_factory):
        input_cloudvolume = cloudvolume_factory.create('input')
        output_cloudvolume = cloudvolume_factory.create('output')
        output_cloudvolume_final = cloudvolume_factory.create('output_final')

        CloudVolumeDatasourceManager(input_cloudvolume, output_cloudvolume=output_cloudvolume,
                                     output_cloudvolume_final=output_cloudvolume_final)

    def test_create_without_final(self, cloudvolume_factory):
        input_cloudvolume = cloudvolume_factory.create('input')
        output_cloudvolume = cloudvolume_factory.create('output')

        CloudVolumeDatasourceManager(input_cloudvolume, output_cloudvolume=output_cloudvolume)

    def test_create_with_final_wrong_cloudvolume(self, cloudvolume_factory):
        input_cloudvolume = cloudvolume_factory.create('input')
        output_cloudvolume = cloudvolume_factory.create('output')
        output_cloudvolume_final = cloudvolume_factory.create('output_final', cloudvolume_class=CloudVolume)

        with pytest.raises(ValueError):
            CloudVolumeDatasourceManager(input_cloudvolume, output_cloudvolume=output_cloudvolume,
                                         output_cloudvolume_final=output_cloudvolume_final)

    def test_fail_not_cloudvolumeczyx(self, cloudvolume_factory):
        input_cloudvolume = cloudvolume_factory.create('input', cloudvolume_class=CloudVolume)
        output_cloudvolume = cloudvolume_factory.create('output', cloudvolume_class=CloudVolume)

        with pytest.raises(ValueError):
            CloudVolumeDatasourceManager(input_cloudvolume, output_cloudvolume)

    def test_create_mod_index(self, cloudvolume_factory):
        input_cloudvolume = cloudvolume_factory.create('input')
        output_cloudvolume = cloudvolume_factory.create('output')
        output_cloudvolume_overlap = cloudvolume_factory.create(
            default_overlap_name(output_cloudvolume, (0,) * len(input_cloudvolume.volume_size)))

        repository = CloudVolumeDatasourceManager(input_cloudvolume, output_cloudvolume, output_cloudvolume_overlap)

        datasource = repository.overlap_repository.get_datasource((10, 11, 12))

        assert repository.overlap_repository.get_datasource((1, 2, 0)) == datasource

    def test_pickle(self, input_cloudvolume):
        from google.auth.exceptions import DefaultCredentialsError
        try:
            cv = CloudVolumeCZYX('gs://seunglab-test/pinky40_v11/image_rechunked')

            from concurrent.futures import ProcessPoolExecutor, as_completed
            futures = []
            with ProcessPoolExecutor(max_workers=4) as ppe:
                for i in range(0, 5):
                    futures.append(ppe.submit(cv.refresh_info))

                for future in as_completed(futures):
                    # an error should be re-raised in one of the futures
                    future.result()
        except DefaultCredentialsError:
            print('Skipping test because of missing credentials')

    def test_flush(self, chunk_datasource_manager):
        chunk_datasource_manager.buffer_generator = create_buffered_cloudvolumeCZYX
        bounds = (slice(200, 206), slice(100, 106), slice(50, 56))
        chunk_shape = (3, 3, 3)
        block = Block(bounds=bounds, chunk_shape=chunk_shape)

        output_cloudvolume = chunk_datasource_manager.output_datasource

        for chunk in block.chunk_iterator():
            chunk.data = GlobalOffsetArray(np.ones((output_cloudvolume.num_channels,) + chunk_shape),
                                           global_offset=(0,) + tuple(s.start for s in chunk.slices))
            chunk_datasource_manager.dump_chunk(chunk, datasource=output_cloudvolume)

        assert 0 == output_cloudvolume[bounds].sum()

        cv_chunk_shape = output_cloudvolume.underlying[::-1]
        cv_offset = output_cloudvolume.voxel_offset[::-1]
        cv_size = output_cloudvolume.volume_size[::-1]

        cv_bounds = tuple(slice(o, o + s) for o, s in zip(cv_offset, cv_size))
        cv_block = Block(bounds=cv_bounds, chunk_shape=cv_chunk_shape)

        for chunk in cv_block.chunk_iterator():
            chunk_datasource_manager.flush(chunk, output_cloudvolume)

        assert np.product(block.shape) * output_cloudvolume.num_channels == output_cloudvolume[bounds].sum()
