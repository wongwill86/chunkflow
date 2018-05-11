class InferenceEngine(object):
    def __init__(self, source):
        self.source = source

    def process(self, bounds):
        pass


class IdentityInference(InferenceEngine):
    def __init__(self, factor, *args, **kwargs):
        super().__init__(args, kwargs)
        self.factor = factor

    def run_inference(self, bounds):
        self.source[bounds] *= self.factor
