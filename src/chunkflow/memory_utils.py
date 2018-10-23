import psutil

def get_memory_uss(process):
    # use uss to take account of shared library memory
    try:
        return process.memory_full_info().uss / 2. ** 30
    except psutil._exceptions.NoSuchProcess:
        return 0

def get_memory_pss(process):
    # use uss to take account of shared library memory
    try:
        return process.memory_full_info().pss / 2. ** 30
    except psutil._exceptions.NoSuchProcess:
        return 0

def print_memory(process):
    processes = [process] + process.children(recursive=True)
    total_memory_pss = sum(map(get_memory_pss, processes))
    total_memory_uss = sum(map(get_memory_uss, processes))
    import objgraph
    print('Total pss: %.3f' % total_memory_pss, 'Total uss: %.3f' % total_memory_uss)
    memory_used = 0
    return total_memory_pss, total_memory_uss
