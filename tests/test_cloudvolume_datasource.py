import unittest
import pytest
import numpy as np
from cloudvolume import CloudVolume
from chunkflow.cloudvolume_datasource import CloudVolumeCZYX
from chunkflow.cloudvolume_datasource import CloudVolumeDatasource


class CloudVolumeCZYXTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def init_dir(self, tmpdir):
        tmpdir.chdir()

    def test_get(self):
        folder = 'file://'
        data_shape_fortran = [8, 8, 4]
        slices_fortran = tuple(slice(0, d) for d in data_shape_fortran)
        slices_c = slices_fortran[::-1]

        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type='image',
            data_type='uint16',
            encoding='raw',
            chunk_size=data_shape_fortran,
            volume_size=data_shape_fortran,
            resolution=[1, 1, 1],
            voxel_offset=[0, 0, 0]
        )
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
            for z in range(0, data_shape_fortran[2]):
                for y in range(0, data_shape_fortran[1]):
                    for x in range(0, data_shape_fortran[0]):
                        self.assertEquals(expected[x][y][z][c], actual[c][z][y][x])

    def test_set(self):
        folder = 'file://'
        data_shape_fortran = [8, 8, 4]
        data_shape_c = data_shape_fortran[::-1]
        slices_fortran = tuple(slice(0, d) for d in data_shape_fortran)
        slices_c = slices_fortran[::-1]

        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type='image',
            data_type='uint16',
            encoding='raw',
            chunk_size=data_shape_fortran,
            volume_size=data_shape_fortran,
            resolution=[1, 1, 1],
            voxel_offset=[0, 0, 0]
        )
        cv_c = CloudVolumeCZYX(folder, info=info, non_aligned_writes=True, fill_missing=True, compress=False)
        cv_c.commit_info()

        # C ordered czyx
        data_c = np.arange(0, np.product(data_shape_c), dtype=np.uint16).reshape(
            data_shape_c, order='C')
        cv_c[slices_c] = data_c
        data_c = np.expand_dims(data_c, 0)
        # expected = cv_fortran[slices_fortran]

        cv_fortran = CloudVolume(folder, info=info, non_aligned_writes=True, fill_missing=True, compress=False)
        actual = cv_fortran[slices_fortran]

        for c in range(0, 1):
            for z in range(0, data_shape_fortran[2]):
                for y in range(0, data_shape_fortran[1]):
                    for x in range(0, data_shape_fortran[0]):
                        self.assertEquals(data_c[c][z][y][x], actual[x][y][z][c])


class CloudVolumeDatasourceTest(unittest.TestCase):

    def test_nothing(self):
        return
        input_cloudvolume = CloudVolumeCZYX('gs://wwong/sub_pinky40_v11/image/', cache=True)
        # cloudvolume_datasource = CloudVolumeDatasource(input_cloudvolume)
        data = input_cloudvolume[0:16, 40960:40960+64, 10240:10240+64]
        print(data.shape)
        print(data.flags)
