from functools import reduce
import inspect
import os

import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from cloudvolume import CloudVolume
from cloudvolume.storage import reset_connection_pools

from chunkflow.datasource_manager import DatasourceRepository

OVERLAP_POSTFIX = '_overlap%s/'
CLOUDVOLUME_INIT_ARGS = len(inspect.getargspec(CloudVolume.__init__)[0]) - 1 # -1 for self arg


def get_index_name(index):
    return reduce(lambda x, y: x + '_' + str(y), index, '')


def default_overlap_name(path_or_cv, mod_index):
    if isinstance(path_or_cv, CloudVolume):
        layer_cloudpath = path_or_cv.layer_cloudpath
    else:
        layer_cloudpath = path_or_cv

    index_name = get_index_name(mod_index)

    if layer_cloudpath.endswith('/'):
        return layer_cloudpath[:-1] + OVERLAP_POSTFIX % index_name
    else:
        return layer_cloudpath + OVERLAP_POSTFIX % index_name


def default_overlap_datasource(path_or_cv, mod_index):
    return CloudVolumeCZYX(default_overlap_name(path_or_cv, mod_index), cache=False, non_aligned_writes=True,
                           fill_missing=True)


class CloudVolumeCZYX(CloudVolume):
    """
    Cloudvolume assumes XYZC Fortran order.  This class hijacks cloud volume indexing to use CZYX C order indexing
    instead. All other usages of indices such as in the constructor are STILL in ZYXC fortran order!!!!
    """
    def __init__(self, *args, **kwargs):
        if len(args) > CLOUDVOLUME_INIT_ARGS:
            args = args[:-1]
            if args[-1] is not os.getpid():
                reset_connection_pools()
        super().__init__(*args, **kwargs)

    def __reduce__(self):
        """
        Help make pickle serialization much easier
        """
        return (
            CloudVolumeCZYX,
            (
                self.layer_cloudpath, self.mip, self.bounded, self.autocrop, self.fill_missing, self.cache,
                self.cdn_cache, self.progress, self.info, None, self.compress, self.non_aligned_writes, self.parallel,
                self.output_to_shared_memory, os.getpid()
            ),
        )

    def __getitem__(self, slices):
        # convert this from Fortran xyzc order because offset is kept in czyx c-order for this class
        dataset_offset = tuple(self.voxel_offset[::-1])
        dataset_offset = (0,) * (len(slices) - len(dataset_offset)) + dataset_offset

        offset = tuple(d_offset if s.start is None else s.start
                       for d_offset, s in zip(dataset_offset, slices) if isinstance(s, slice))

        # reverse to Fortran xyzc order
        slices = slices[::-1]

        item = super().__getitem__(slices)
        if hasattr(item, 'flags') and (item.flags['F_CONTIGUOUS'] or not item.flags['C_CONTIGUOUS']):
            item = item.transpose()
            item = np.ascontiguousarray(item)

        # ugh cloudvolume always adds dimension layer
        if len(item.shape) > len(offset):
            offset = (0,) * (len(item.shape) - len(offset)) + offset

        arr = GlobalOffsetArray(item, global_offset=offset)
        return arr

    def __setitem__(self, slices, item):
        slices = slices[::-1]
        if hasattr(item, 'flags') and (not item.flags['F_CONTIGUOUS'] or item.flags['C_CONTIGUOUS']):
            item = item.transpose()
            item = np.asfortranarray(item)
        super().__setitem__(slices, item)


class CloudVolumeDatasourceRepository(DatasourceRepository):
    def __init__(self, input_cloudvolume, output_cloudvolume, output_cloudvolume_final=None,
                 overlap_protocol=None, *args, **kwargs):
        if overlap_protocol is None:
            self.overlap_protocol = output_cloudvolume.path.protocol + '://'
        else:
            self.overlap_protocol = overlap_protocol

        if any(not isinstance(volume, CloudVolumeCZYX) for volume in [input_cloudvolume, output_cloudvolume]) or \
                (output_cloudvolume_final is not None and not isinstance(output_cloudvolume_final, CloudVolumeCZYX)):
            raise ValueError('Must use %s class cloudvolume to ensure correct c order indexing' %
                             CloudVolumeCZYX.__name__)
        super().__init__(input_datasource=input_cloudvolume,
                         output_datasource=output_cloudvolume,
                         output_datasource_final=output_cloudvolume_final,
                         *args, **kwargs)

    def create(self, mod_index, *args, **kwargs):
        layer_cloudpath = default_overlap_name(self.output_datasource, mod_index)
        post_protocol_index = layer_cloudpath.find("//") + 2
        base_name = layer_cloudpath[post_protocol_index:]
        layer_cloudpath = self.overlap_protocol + base_name

        try:
            new_cloudvolume = CloudVolumeCZYX(layer_cloudpath, cache=False, non_aligned_writes=True, fill_missing=True)
        except ValueError:
            base_info = self.output_datasource.info
            new_cloudvolume = CloudVolumeCZYX(layer_cloudpath, info=base_info, cache=False, non_aligned_writes=True,
                                              fill_missing=True)
            new_cloudvolume.commit_info()

        return new_cloudvolume
