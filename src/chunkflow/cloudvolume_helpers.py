import fractions
import functools
import math

from cloudvolume import CloudVolume

from chunkflow.cloudvolume_datasource import CloudVolumeCZYX


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
    gcd = functools.reduce(fractions.gcd, core_shape)
    factors = get_factors(gcd)

    chunk_shape_options = [tuple(cs // factor for cs in core_shape) for factor in factors]

    return chunk_shape_options


def valid_cloudvolume(path_or_cv, chunk_shape_options):
    try:
        if isinstance(path_or_cv, CloudVolume):
            cloudvolume = path_or_cv
        else:
            cloudvolume = CloudVolumeCZYX(path_or_cv)

        actual_chunk_size = tuple(cloudvolume.underlying)

        if not any(tuple(actual_chunk_size) == chunk_shape for chunk_shape in chunk_shape_options):
            print('Warning: %s already has incorrect chunk size %s. Please reformat with one of these chunk sizes: %s' %
                  (cloudvolume.layer_cloudpath, actual_chunk_size, chunk_shape_options))
            return False
    except ValueError:
        print('Warning: %s does not exist!' % (path_or_cv,))
        return False
    return True
