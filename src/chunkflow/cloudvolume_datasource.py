from functools import reduce

import numpy as np

from chunkflow.datasource_manager import DatasourceRepository
from chunkflow.global_offset_array import GlobalOffsetArray
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


class CloudVolumeDatasourceRepository(DatasourceRepository):
    def __init__(self, input_cloudvolume, output_cloudvolume_core, output_cloudvolume_overlap,
                 intermediate_protocol='file://', *args, **kwargs):
        self.intermediate_protocol = intermediate_protocol
        # input_datasource = CloudVolumeWrapper(input_cloudvolume)
        if any(not isinstance(volume, CloudVolumeCZYX)
               for volume in [input_cloudvolume, output_cloudvolume_core, output_cloudvolume_overlap]):
            raise ValueError('Must use %s class cloudvolume to ensure correct c order indexing' %
                             CloudVolumeCZYX.__name__)
        super().__init__(input_cloudvolume,
                         output_cloudvolume_core,
                         output_cloudvolume_overlap,
                         *args, **kwargs)

    def create(self, mod_index, *args, **kwargs):
        post_protocol_index = self.output_datasource_core.layer_cloudpath.find("//") + 2
        base_name = self.output_datasource_core.layer_cloudpath[post_protocol_index:]
        base_info = self.output_datasource_core.info
        index_name = reduce(lambda x, y: x + '_' + str(y), mod_index, '')
        new_cloudvolume = CloudVolume(self.intermediate_protocol + base_name + index_name,
                                      info=base_info, cache=False, non_aligned_writes=True, fill_missing=True,
                                      compress=False)
        new_cloudvolume.commit_info()
        return new_cloudvolume
