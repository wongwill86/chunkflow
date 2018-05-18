class InferenceEngine(object):
    def _process(self, data):
        """
        returns processed data
        """
        raise NotImplemented

    def run_inference(self, data):
        return self._process(data)


class IdentityInference(InferenceEngine):
    def __init__(self, factor=1, *args, **kwargs):
        self.factor = factor

    def _process(self, data):
        return data * self.factor
