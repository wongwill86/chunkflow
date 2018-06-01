import numpy as np
from chunkflow.global_offset_array import GlobalOffsetArray
from chunkflow.datasource_manager import DatasourceRepository
from cloudvolume import CloudVolume

class CloudVolumeWrapper(CloudVolume):
    def __getitem__(self, slices):
        print(slices)
        offset = [s.start for s in slices if isinstance(s, slice)]
        print(offset)
        item = super().__getitem__(slices);
        if item.flags['F_CONTIGUOUS']:
            item = item.transpose()
            item = np.ascontiguousarray(item)

        GlobalOffsetArray(super().__getitem__(slices), global_offset=offset)


class CloudVolumeDatasource(DatasourceRepository):
    def __init__(self, input_cloudvolume, *args, **kwargs):
        # input_datasource = CloudVolumeWrapper(input_cloudvolume)
        super().__init__(input_datasource, *args, **kwargs)

    def create(self, mod_index, *args, **kwargs):
        offset = self.input_datasource.global_offset
        shape = self.input_datasource.shape

        return GlobalOffsetArray(np.zeros(shape), global_offset=offset)

