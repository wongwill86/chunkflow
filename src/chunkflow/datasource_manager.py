import numpy as np
from chunkflow.iterators import UnitIterator
from chunkflow.global_offset_array import GlobalOffsetArray

def get_mod_index(index):
    return tuple(idx % 3 for idx in index)

class DatasourceManager(object):
    def __init__(self, input_datasource):
        self.repository = dict()
        self.input_datasource = input_datasource
        self.output_datasource_core = None
        self.output_datasource_edge = None
        iterator = UnitIterator()

        origin = (1,1)
        self.repository[origin] = self.create(origin)
        for neighbor in iterator.get_all_neighbors(origin):
            self.repository[neighbor] = self.create(neighbor)



    def create(self, mod_index, *args, **kwargs):
        raise NotImplemented

    def get_datasource(self, index):
        mod_index = get_mod_index(index)
        # if index not in self.repository:
        #     self.repository[mod_index] = self.create(mod_index)

        return self.repository[mod_index]

class NumpyDatasource(DatasourceManager):
    # def __init__(self, input_datasource):
    #     super().__init__(input_datasource)
    def create(self, mod_index, *args, **kwargs):
        return GlobalOffsetArray(np.ones((100,100)), global_offset=(0,0))
