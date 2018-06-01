import unittest
import cloudvolume
# from cloudvolume import CloudVolume
from chunkflow.cloudvolume_datasource import CloudVolumeWrapper
from chunkflow.cloudvolume_datasource import CloudVolumeDatasource

class CloudVolumeDatasourceTest(unittest.TestCase):

    def test_nothing(self):
        input_cloudvolume = CloudVolumeWrapper('gs://wwong/sub_pinky40_v11/image/', cache=True)
        # cloudvolume_datasource = CloudVolumeDatasource(input_cloudvolume)
        data = input_cloudvolume[10240:10240+64, 40960:40960+64,0:16]
        print(data.shape)
        print(data.flags)


        assert False
