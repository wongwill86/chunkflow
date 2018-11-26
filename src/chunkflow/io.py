import importlib
import os
import tarfile
import tempfile
import types

from cloudvolume.storage import Storage


def download_to_local(remote_location):
    last_slash = remote_location.rfind('/')
    assert last_slash > -1, 'Unable to parse folder location from remote_location %s' % remote_location

    remote_folder = remote_location[:last_slash]
    remote_filename = remote_location[last_slash + 1:]
    remote_storage = Storage(remote_folder)

    temp_dir = tempfile.mkdtemp()
    local_location = os.path.join(temp_dir, remote_filename)

    with open(local_location, 'wb') as f:
        print('Downloading %s to %s' % (remote_location, local_location))
        downloaded_file = remote_storage.get_file(remote_filename)
        if downloaded_file is None:
            raise IOError('remote_location %s not found' % remote_location)
        f.write(downloaded_file)

    if tarfile.is_tarfile(local_location):
        directories = [d for d in tarfile.getmembers() if d.isdir()]
        tarfile.extractall(temp_dir)

        if len(directories) == 1:
            local_location = os.path.join(local_location, directories[0].name)
        else:
            assert len(directories) == 0, 'Only one directory should be given in zip, found: %s' % directories

    return local_location


def load_source(fname, module_name="Model"):
    """ Imports a module from source """
    loader = importlib.machinery.SourceFileLoader(module_name, fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod
