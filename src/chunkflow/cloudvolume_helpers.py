import fractions
import functools
import math

from cloudvolume import CloudVolume

from chunkflow.cloudvolume_datasource import CloudVolumeCZYX

ATTRIBUTE_COMPARISONS = ['voxel_offset', 'volume_size']
TEMPLATE_INFO_ARGS = {
    'layer_type': 'image',
    'encoding': 'raw',
    'resolution': [1, 1, 1],
}
TEMPLATE_ARGS = [
    'data_type',
    'volume_size',
    'voxel_offset',
    'num_channels'
]


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


def valid_cloudvolume(path_or_cv, chunk_shape_options, input_datasource):
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

        for attribute in ATTRIBUTE_COMPARISONS:
            print('attribute', attribute)
            print(getattr(input_datasource, attribute))
            print(getattr(cloudvolume, attribute))
            if str(getattr(input_datasource, attribute)) != str(getattr(cloudvolume, attribute)):
                print('Warning: %s already has incorrect property %s compared to %s with %s.' % (
                          cloudvolume.layer_cloudpath, getattr(input_datasource, attribute),
                          input_datasource.layer_cloudpath, getattr(input_datasource, attribute)
                      ))
                return False

    except ValueError as ve:
        print('Warning: %s does not exist! %s' % (path_or_cv, ve))
        return False
    return True


def create_cloudvolume(layer_cloudpath, chunk_size, input_datasource, **kwargs):
    print(kwargs)
    info_args = TEMPLATE_INFO_ARGS.copy()
    info_args['resolution'] = input_datasource.resolution
    info_args['chunk_size'] = chunk_size

    for argument in TEMPLATE_ARGS:
        if argument not in kwargs or kwargs[argument] is None:
            info_args[argument] = getattr(input_datasource, argument)
        else:
            info_args[argument] = kwargs[argument]

    info = CloudVolume.create_new_info(**info_args)

    cloudvolume = CloudVolumeCZYX(layer_cloudpath, info=info)
    cloudvolume.commit_info()
    print('Created ', layer_cloudpath)
