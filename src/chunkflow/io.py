import importlib
import os
import types

from cloudvolume.storage import Storage


def download_to_local(remote_location):
    last_slash = remote_location.rfind('/')
    assert last_slash > -1, 'Unable to parse folder location from remote_location %s' % remote_location

    remote_folder = remote_location[:last_slash]
    remote_filename = remote_location[last_slash + 1:]
    remote_storage = Storage(remote_folder)

    local_location = os.path.join('/tmp', remote_filename)

    with open(local_location, 'wb') as f:
        print('Downloading %s to %s' % (remote_location, local_location))
        downloaded_file = remote_storage.get_file(remote_filename)
        if downloaded_file is None:
            raise IOError('remote_location %s not found' % remote_location)
        f.write(downloaded_file)

    return local_location


def load_source(fname, module_name="Model"):
    """ Imports a module from source """
    loader = importlib.machinery.SourceFileLoader(module_name, fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod
