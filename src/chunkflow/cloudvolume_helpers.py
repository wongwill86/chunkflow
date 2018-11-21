import math

from cloudvolume import CloudVolume

from chunkflow.cloudvolume_datasource import CloudVolumeCZYX

ATTRIBUTE_COMPARISONS = ['voxel_offset', 'volume_size']
TEMPLATE_INFO_ARGS = {
    'layer_type': 'image',
    'encoding': 'raw',
    'resolution': [1, 1, 1],
}
# dict from argument name to whether or not the value should be reversed
TEMPLATE_ARGS = {
    'data_type': False,
    'volume_size': True,
    'voxel_offset': True,
    'num_channels': False
}


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


def get_possible_chunk_sizes(overlap, task_shape, min_mips, num_patches):
    core_shape = tuple(p - o for p, o in zip(task_shape, overlap))

    min_mip_factor = 2 ** min_mips
    assert all(c_s % min_mip_factor == 0 for c_s in core_shape), 'Unable to support %s with %s mips, core shape %s' % (
        task_shape, min_mips, core_shape)
    core_shape = tuple(c_s // min_mip_factor * n_p for c_s, n_p in zip(core_shape, num_patches))

    chunk_shape_options = [
        [c_s / i * min_mip_factor for i in range(1, c_s) if c_s % i == 0] for c_s in core_shape
    ]

    return chunk_shape_options


def valid_cloudvolume(path_or_cv, chunk_shape_options, input_datasource):
    try:
        if isinstance(path_or_cv, CloudVolume):
            cloudvolume = path_or_cv
        else:
            cloudvolume = CloudVolumeCZYX(path_or_cv)

        # cloudvolume is in xyzc
        actual_chunk_size = tuple(cloudvolume.underlying)[::-1]

        for c_size, chunk_shape_option in zip(actual_chunk_size, chunk_shape_options):
            if c_size not in chunk_shape_option:
                print('Warning: %s has incorrect chunk size %s. These chunk sizes are compatible: %s' % (
                    cloudvolume.layer_cloudpath, actual_chunk_size, chunk_shape_options))
                return False

        for attribute in ATTRIBUTE_COMPARISONS:
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
    info_args = TEMPLATE_INFO_ARGS.copy()
    info_args['resolution'] = input_datasource.resolution

    # convert back to xyzc for cloudvolume
    info_args['chunk_size'] = chunk_size[::-1]

    for argument in TEMPLATE_ARGS:
        if argument not in kwargs or kwargs[argument] is None:
            info_args[argument] = getattr(input_datasource, argument)
        else:
            info_args[argument] = kwargs[argument]
            if TEMPLATE_ARGS[argument]:
                # convert back to xyzc for cloudvolume
                info_args[argument] = info_args[argument][::-1]

    info = CloudVolume.create_new_info(**info_args)

    cloudvolume = CloudVolumeCZYX(layer_cloudpath, info=info)
    cloudvolume.commit_info()
    print('Created ', layer_cloudpath)
