import unittest
import pytest
import numpy as np
from cloudvolume import CloudVolume
from chunkflow.cloudvolume_datasource import CloudVolumeCZYX
from chunkflow.cloudvolume_datasource import CloudVolumeDatasource


class CloudVolumeCZYXTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def test_get(self, tmpdir_factory):
        folder = 'file://' + tmpdir_factory.mktemp('test_get').strpath
        data_shape_fortran = [4, 4, 1]
        slices_fortran = tuple(slice(0, d) for d in data_shape_fortran)
        slices_c = slices_fortran[::-1]

        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type='image',
            data_type='uint8',
            encoding='raw',
            chunk_size=data_shape_fortran,
            volume_size=data_shape_fortran,
            resolution=[1, 1, 1],
            voxel_offset=[0, 0, 0]
        )
        cv_fortran = CloudVolume(folder, info=info, non_aligned_writes=True, fill_missing=True, compress=False)
        cv_fortran.commit_info()

        # F ordered czyx
        data_fortran = np.arange(0, np.product(data_shape_fortran), dtype=np.uint8).reshape(
            data_shape_fortran, order='F')
        cv_fortran[slices_fortran] = data_fortran
        expected = cv_fortran[slices_fortran]
        print(expected.flatten())

        cv_c = CloudVolumeCZYX(folder, info=info, non_aligned_writes=True, fill_missing=True, compress=False)
        actual = cv_c[slices_c].flatten()
        import binascii
        print(binascii.b2a_hex(actual.tobytes()))
        print(actual)
        print(folder)

        assert False


class CloudVolumeDatasourceTest(unittest.TestCase):

    def test_nothing(self):
        input_cloudvolume = CloudVolumeCZYX('gs://wwong/sub_pinky40_v11/image/', cache=True)
        # cloudvolume_datasource = CloudVolumeDatasource(input_cloudvolume)
        data = input_cloudvolume[0:16, 40960:40960+64, 10240:10240+64]
        print(data.shape)
        print(data.flags)

        assert False
