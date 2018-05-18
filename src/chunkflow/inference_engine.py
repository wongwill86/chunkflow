import time
from threading import current_thread


class InferenceEngine(object):
    def _process(self, data):
        """
        returns processed data
        """
        raise NotImplemented

    def run_inference(self, data):
        # self.source[bounds] *= self.factor
        return self._process(data)

class IdentityInference(InferenceEngine):
    def __init__(self, factor=1, *args, **kwargs):
        self.factor = factor

    def _process(self, data):
        # print('>>>>>> %s ruhnning inference!! %s' % (current_thread().name, data.shape))
        # time.sleep(1)
        return data * self.factor
            # .flat_map(lambda index:
            #           (
            #               Observable.combine_latest(
            #                   Observable.just(index).map(index_to_slices_partial),
            #                   Observable.just(self.datasource_manager.input_datasource),
            #                   Observable.just(index).map(self.datasource_manager.get_datasource),
            #                   lambda slices, input_datasource, output_datasource:
            #                   output_datasource.__setitem__(slices, input_datasource[slices])
            #               )
            #               Observable.just(index).zip(
            #                   index_to_slices_partial(index),
            #                   self.datasource_manager.input_datasource,
            #                   self.datasource_manager.get_datasource(index)
            #               )
            #               .map(lambda data: index)
            #           )
