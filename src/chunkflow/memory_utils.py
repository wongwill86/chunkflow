import psutil


def get_memory_usage(process, memory_type='pss'):
    try:
        memory = getattr(process.memory_full_info(), memory_type) / 2. ** 30
        print(process, ' is using ', memory, 'GiB', memory_type)
        return memory
    except psutil._exceptions.NoSuchProcess:
        return 0


def get_memory_usage_pss(process):
    return get_memory_usage(process, 'pss')


def get_memory_usage_uss(process):
    uss = get_memory_usage(process, 'uss')
    return uss


def print_memory(process):
    processes = [process] + process.children(recursive=True)
    memory_rss = get_memory_usage(psutil.Process(), memory_type='rss')
    memory_uss = sum(map(get_memory_usage_uss, processes))
    memory_pss = sum(map(get_memory_usage_pss, processes))
    print('Total rss: %.3f GiB, Total pss: %.3f GiB, Total uss %.3f' % (memory_rss, memory_pss, memory_uss))
    return memory_pss
