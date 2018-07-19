from functools import reduce

import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from cloudvolume import CloudVolume, txrx
from cloudvolume.lib import Bbox, generate_slices, Vec, extract_path

from chunkflow.datasource_manager import DatasourceRepository

OVERLAP_POSTFIX = '_overlap%s/'


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

    def __reduce__(self):
        return (
            CloudVolumeCZYX,
            (
                self.layer_cloudpath, self.mip, self.bounded, self.autocrop, self.fill_missing, self.cache,
                self.cdn_cache, self.progress, self.info, None, self.compress, self.non_aligned_writes, self.parallel,
                self.output_to_shared_memory
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

        # replace this thing here!!
        # self._get_slices(slices)
        item = super().__getitem__(slices)

        if hasattr(item, 'flags') and (item.flags['F_CONTIGUOUS'] or not item.flags['C_CONTIGUOUS']):
            item = item.transpose()
            item = np.ascontiguousarray(item)

        # ugh cloudvolume always adds dimension layer
        if len(item.shape) > len(offset):
            offset = (0,) * (len(item.shape) - len(offset)) + offset

        arr = GlobalOffsetArray(item, global_offset=offset)
        return arr

    def _get_slices(self, slices):
        '''
        Overriding CV internal get item
        '''
        # stolen from cloudvolume.volume.__getitem__
        (requested_bbox, steps, channel_slice) = self.__interpret_slices(slices)
        # stolen from cloudvolume.txrx.cutout
        cloudpath_bbox = requested_bbox.expand_to_chunk_size(self.underlying, offset=self.voxel_offset)
        cloudpath_bbox = Bbox.clamp(cloudpath_bbox, self.bounds)
        cloudpaths = txrx.chunknames(cloudpath_bbox, self.bounds, self.key, self.underlying)
        for cloudpath in cloudpaths:
            print(cloudpath, ' ', self.get_path_to_file(cloudpath))
        # locations = self.cache.compute_data_locations(cloudpaths)
        # for location, vals in locations.items():
        #     for val in vals:
        #         print(location, val)

        layer_path = self.layer_cloudpath
        extracted_path = extract_path(layer_path)
        print(extracted_path)

    def get_path_to_file(self, cloudpath):
        path = extract_path(self.layer_cloudpath)
        return '/'.join([path.intermediate_path, path.dataset, path.layer, cloudpath])

    def __interpret_slices(self, slices):
        """
        stolen from CV becaues it was private
        Convert python slice objects into a more useful and computable form:

        - requested_bbox: A bounding box representing the volume requested
        - steps: the requested stride over x,y,z
        - channel_slice: A python slice object over the channel dimension

        Returned as a tuple: (requested_bbox, steps, channel_slice)
        """
        maxsize = list(self.bounds.maxpt) + [self.num_channels]
        minsize = list(self.bounds.minpt) + [0]

        slices = generate_slices(slices, minsize, maxsize, bounded=self.bounded)
        channel_slice = slices.pop()

        minpt = Vec(*[slc.start for slc in slices])
        maxpt = Vec(*[slc.stop for slc in slices])
        steps = Vec(*[slc.step for slc in slices])

        return Bbox(minpt, maxpt), steps, channel_slice

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
            new_cloudvolume = CloudVolumeCZYX(layer_cloudpath, cache=False, non_aligned_writes=True, fill_missing=True,
                                              compress=False)
        except ValueError:
            base_info = self.output_datasource.info
            new_cloudvolume = CloudVolumeCZYX(layer_cloudpath, info=base_info, cache=False, non_aligned_writes=True,
                                              fill_missing=True, compress=False)
            new_cloudvolume.commit_info()

        return new_cloudvolume
