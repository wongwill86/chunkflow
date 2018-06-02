import numpy as np
from chunkflow.global_offset_array import GlobalOffsetArray
from chunkflow.datasource_manager import DatasourceRepository
from cloudvolume import CloudVolume


class CloudVolumeCZYX(CloudVolume):
    """
    Cloudvolume assumes XYZC Fortran order.  This class hijacks cloud volume indexing to use CZYX C order indexing
    instead. All other usages of indices such as in the constructor are STILL in ZYXC fortran order!!!!
    """

    def __getitem__(self, slices):
        offset = tuple(s.start for s in slices if isinstance(s, slice))

        # reverse to Fortran xyzc order
        slices = slices[::-1]

        item = super().__getitem__(slices)
        if item.flags['F_CONTIGUOUS']:
            item = item.transpose()
            item = np.ascontiguousarray(item)

        # ugh cloudvolume always adds dimension layer
        if len(item.shape) > len(offset):
            offset = (0,) * (len(item.shape) - len(offset)) + offset

        arr = GlobalOffsetArray(item, global_offset=offset)
        return arr

    def __setitem__(self, slices, item):
        slices = slices[::-1]
        if item.flags['C_CONTIGUOUS']:
            item = item.transpose()
            item = np.asfortranarray(item)
        super().__setitem__(slices, item)


class CloudVolumeDatasource(DatasourceRepository):
    def __init__(self, input_cloudvolume, *args, **kwargs):
        # input_datasource = CloudVolumeWrapper(input_cloudvolume)
        if not isinstance(input_cloudvolume, CloudVolumeCZYX):
            raise NotImplementedError('Must use %s class cloudvolume to ensure correct indexing' %
                                      CloudVolumeCZYX.__class__)
        super().__init__(input_cloudvolume, *args, **kwargs)

    def create(self, mod_index, *args, **kwargs):
        pass
