class InferenceEngine(object):
    def _process(self, data):
        """
        returns processed data
        """
        raise NotImplemented

    def run_inference(self, input_datasource, bounds):
        # self.source[bounds] *= self.factor
        # print('-------------- running identity inference on %s' % (bounds,))
        return self._process(input_datasource[bounds])


class IdentityInference(InferenceEngine):
    def __init__(self, factor=1, *args, **kwargs):
        self.factor = factor

    def _process(self, data):
        return data * self.factor


