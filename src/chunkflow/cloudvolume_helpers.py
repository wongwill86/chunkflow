import functools
import math

from cloudvolume import CloudVolume

from chunkflow.cloudvolume_datasource import CloudVolumeCZYX

OVERLAP_POSTFIX = '_overlap/'


def get_factors(n):
    factors = []
    idx = 1
    other = n
    while idx <= math.sqrt(n) and idx < other:
        if n % idx == 0:
            factors.append(idx)
            factors.append(n // idx)
        idx += 1

    factors.sort()
    return factors


def get_possible_chunk_sizes(overlap, patch_shape):
    core_shape = tuple(p - o for p, o in zip(patch_shape, overlap))
    gcd = functools.reduce(math.gcd, core_shape)
    factors = get_factors(gcd)

    chunk_shape_options = [tuple(cs / factor for cs in core_shape) for factor in factors]

    return chunk_shape_options


def valid_cloudvolume(path_or_cv, chunk_shape_options):
    try:
        if isinstance(path_or_cv, CloudVolume):
            cloudvolume = path_or_cv
        else:
            cloudvolume = CloudVolumeCZYX(path_or_cv, cache=False, non_aligned_writes=True, fill_missing=True)

        actual_chunk_size = cloudvolume.info['scales'][cloudvolume.mip]['chunk_sizes']

        if not any(tuple(actual_chunk_size) == chunk_shape for chunk_shape in chunk_shape_options):
            print('Warning: %s already has incorrect chunk size %s. Please reformat with one of these chunk sizes: %s' %
                  (cloudvolume.layer_cloudpath, actual_chunk_size, chunk_shape_options))
            return False
    except ValueError:
        print(path_or_cv, ' does not exist!')
        return False
    return True


def to_overlap_name(layer_cloudpath):
    if layer_cloudpath.endswith('/'):
        return layer_cloudpath[:-1] + OVERLAP_POSTFIX
    else:
        return layer_cloudpath + OVERLAP_POSTFIX


def get_cloudvolume_overlap(path_or_cv):
    if isinstance(path_or_cv, CloudVolume):
        return CloudVolumeCZYX(
            to_overlap_name(path_or_cv.layer_cloudpath), cache=False, non_aligned_writes=True, fill_missing=True)
    else:
        return CloudVolumeCZYX(to_overlap_name(path_or_cv), cache=False, non_aligned_writes=True, fill_missing=True)
