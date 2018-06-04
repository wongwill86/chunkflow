import unittest
import pytest
import numpy as np
from cloudvolume import CloudVolume
from chunkflow.cloudvolume_datasource import CloudVolumeCZYX
from chunkflow.cloudvolume_datasource import CloudVolumeDatasource
from chunkflow.global_offset_array import GlobalOffsetArray


INPUT_SINGLE_CHANNEL_PATH = 'file://input'
VOXEL_OFFSET = [100, 100, 200]  # this is xyz
DEFAULT_SINGLE_CHANNEL_INFO = CloudVolume.create_new_info(
    num_channels=1,
    layer_type='image',
    data_type='uint16',
    encoding='raw',
    chunk_size=[8, 8, 4],  # this is xyz
    volume_size=[24, 24, 12],  # this is xyz
    resolution=[1, 1, 1],  # this is xyz
    voxel_offset=VOXEL_OFFSET
)

OUTPUT_PATH_CORE = 'file://output_core'
OUTPUT_PATH_OVERLAP = 'file://output_overlap'
DEFAULT_MULTI_CHANNEL_INFO = DEFAULT_SINGLE_CHANNEL_INFO.copy()
DEFAULT_MULTI_CHANNEL_INFO['num_channels'] = 3


@pytest.fixture(scope='function')
def cloud_volume(tmpdir):
    tmpdir.chdir()
    input_cloudvolume = CloudVolume(INPUT_SINGLE_CHANNEL_PATH, info=DEFAULT_SINGLE_CHANNEL_INFO, cache=False)
    input_cloudvolume.commit_info()
    output_cloudvolume_core = CloudVolume(
        OUTPUT_PATH_CORE, info=DEFAULT_MULTI_CHANNEL_INFO, cache=False)
    output_cloudvolume_core.commit_info()
    output_cloudvolume_overlap = CloudVolume(
        OUTPUT_PATH_OVERLAP, info=DEFAULT_MULTI_CHANNEL_INFO, cache=False)
    output_cloudvolume_overlap.commit_info()


@pytest.mark.usefixtures('cloud_volume')
class CloudVolumeCZYXTest(unittest.TestCase):

    def init_dir(self, tmpdir):
        tmpdir.chdir()

    def test_get(self):
        folder = 'file://'
        data_shape_fortran = [8, 8, 4]
        slices_fortran = tuple(slice(0 + o, d + o) for d, o in zip(data_shape_fortran, VOXEL_OFFSET))
        slices_c = slices_fortran[::-1]

        info = DEFAULT_SINGLE_CHANNEL_INFO.copy()
        info['chunk_size'] = data_shape_fortran
        info['volume_size'] = data_shape_fortran

        cv_fortran = CloudVolume(folder, info=info, non_aligned_writes=True, fill_missing=True, compress=False)
        cv_fortran.commit_info()

        # F ordered xyzc
        data_fortran = np.arange(0, np.product(data_shape_fortran), dtype=np.uint16).reshape(
            data_shape_fortran, order='F')
        cv_fortran[slices_fortran] = data_fortran
        expected = cv_fortran[slices_fortran]

        cv_c = CloudVolumeCZYX(folder, info=info, non_aligned_writes=True, fill_missing=True, compress=False)
        actual = cv_c[slices_c]

        for c in range(0, 1):
            for z in range(slices_fortran[2].start, slices_fortran[2].stop):
                for y in range(slices_fortran[1].start, slices_fortran[1].stop):
                    for x in range(slices_fortran[0].start, slices_fortran[0].stop):
                        expected_index = tuple(i - o for i, o in zip((x, y, z, c), VOXEL_OFFSET)) + (0,)
                        self.assertEquals(expected[expected_index], actual[c][z][y][x])

    def test_set(self):
        data_shape_fortran = [8, 8, 4]
        data_shape_c = data_shape_fortran[::-1]
        slices_fortran = tuple(slice(o, d + o) for d, o in zip(data_shape_fortran, VOXEL_OFFSET))
        slices_c = slices_fortran[::-1]

        info = DEFAULT_SINGLE_CHANNEL_INFO.copy()
        info['chunk_size'] = data_shape_fortran
        info['volume_size'] = data_shape_fortran

        cv_c = CloudVolumeCZYX(INPUT_SINGLE_CHANNEL_PATH, info=info, non_aligned_writes=True, fill_missing=True,
                               compress=False)
        cv_c.commit_info()

        # C ordered czyx
        data_c = np.arange(0, np.product(data_shape_c), dtype=np.uint16).reshape(
            data_shape_c, order='C')
        cv_c[slices_c] = data_c
        data_c = np.expand_dims(data_c, 0)
        # expected = cv_fortran[slices_fortran]

        cv_fortran = CloudVolume(INPUT_SINGLE_CHANNEL_PATH, info=info, non_aligned_writes=True, fill_missing=True,
                                 compress=False)
        actual = cv_fortran[slices_fortran]

        for c in range(0, 1):
            for z in range(slices_fortran[2].start, slices_fortran[2].stop):
                for y in range(slices_fortran[1].start, slices_fortran[1].stop):
                    for x in range(slices_fortran[0].start, slices_fortran[0].stop):
                        expected_index = tuple(i - o for i, o in zip((x, y, z), VOXEL_OFFSET)) + (0,)
                        self.assertEquals(data_c[expected_index[::-1]], actual[expected_index])

    def test_multi_dimensional_conversions(self):
        input_cloudvolume = CloudVolumeCZYX(INPUT_SINGLE_CHANNEL_PATH, cache=True, non_aligned_writes=True,
                                            fill_missing=True, compress=False)
        output_cloudvolume_core = CloudVolumeCZYX(OUTPUT_PATH_CORE, cache=True, non_aligned_writes=True,
                                                  fill_missing=True, compress=False)

        sizes = (8, 16, 16)
        slices = tuple(slice(o, s + o) for o, s in zip(VOXEL_OFFSET[::-1], sizes))

        # Setup Data
        input_data = np.arange(0, 8*16*16, dtype='uint16').reshape((8, 16, 16))
        input_cloudvolume[slices] = input_data

        # Pull Data
        input_data = input_cloudvolume[slices]

        global_offset = input_data.global_offset
        one = input_data
        two = input_data * 10
        three = input_data * 100

        output_cloudvolume_core[slices] = GlobalOffsetArray(np.stack((one, two, three)).squeeze(),
                                                            global_offset=global_offset)

        self.assertTrue(np.array_equal(input_data, output_cloudvolume_core[(0,) + slices]))
        self.assertTrue(np.array_equal(input_data * 10, output_cloudvolume_core[(1,) + slices]))
        self.assertTrue(np.array_equal(input_data * 100, output_cloudvolume_core[(2,) + slices]))


@pytest.mark.usefixtures('cloud_volume')
class CloudVolumeDatasourceTest(unittest.TestCase):
    def test_fail_not_cloudvolumeczyx(self):
        input_cloudvolume = CloudVolume(INPUT_SINGLE_CHANNEL_PATH, cache=True)
        output_cloudvolume_core = CloudVolume(INPUT_SINGLE_CHANNEL_PATH, cache=True)
        output_cloudvolume_overlap = CloudVolume(INPUT_SINGLE_CHANNEL_PATH, cache=True)

        # input_cloudvolume = CloudVolumeCZYX('gs://wwong/sub_pinky40_v11/image/', cache=True)
        with self.assertRaises(ValueError):
            CloudVolumeDatasource(input_cloudvolume, output_cloudvolume_core, output_cloudvolume_overlap)

    def test_create(self):
        input_cloudvolume = CloudVolumeCZYX(INPUT_SINGLE_CHANNEL_PATH, cache=True)
        output_cloudvolume_core = CloudVolumeCZYX(INPUT_SINGLE_CHANNEL_PATH, cache=True)
        output_cloudvolume_overlap = CloudVolumeCZYX(INPUT_SINGLE_CHANNEL_PATH, cache=True)

        CloudVolumeDatasource(input_cloudvolume, output_cloudvolume_core, output_cloudvolume_overlap)
    # def test_(self):
    #     input_cloudvolume = CloudVolumeCZYX(INPUT_SINGLE_CHANNEL_PATH, cache=True, non_aligned_writes=True, fill_missing=True,
    #                                         compress=False)
    #     output_cloudvolume_core = CloudVolumeCZYX(self.OUTPUT_PATH_CORE, cache=True, non_aligned_writes=True,
    #                                               fill_missing=True, compress=False)
    #     output_cloudvolume_overlap = CloudVolumeCZYX(self.OUTPUT_PATH_OVERLAP, cache=True, non_aligned_writes=True,
    #                                                  fill_missing=True, compress=False)

    #     # Setup Data
    #     cloudvolume_datasource = CloudVolumeDatasource(input_cloudvolume, output_cloudvolume_core,
    #                                                    output_cloudvolume_overlap)
    #     input_data = np.arange(0, 8*16*16, dtype='uint16').reshape((8, 16, 16))
    #     input_cloudvolume[100:108, 100:116, 100:116] = input_data

    #     # Pull Data
    #     input_data = input_cloudvolume[100:108, 100:116, 100:116]

    #     global_offset = input_data.global_offset
    #     one = input_data
    #     two = input_data * 10
    #     three = input_data * 100
    #     print('go is ')
    #     print(global_offset)

    #     output_cloudvolume_core[100:108, 100:116, 100:116] = GlobalOffsetArray(np.stack((one, two, three)).squeeze(), global_offset=global_offset)
    #     output_cloudvolume_overlap[100:108, 100:116, 100:116] = GlobalOffsetArray(np.stack((one, two, three)).squeeze(), global_offset=global_offset)

    #     self.assertTrue(np.array_equal(input_data, output_cloudvolume_core[0, 100:108, 100:116, 100:116]))
    #     self.assertTrue(np.array_equal(input_data * 10, output_cloudvolume_core[0, 100:108, 100:116, 100:116]))
    #     self.assertTrue(np.array_equal(input_data * 100, output_cloudvolume_core[0, 100:108, 100:116, 100:116]))
        # assert False
