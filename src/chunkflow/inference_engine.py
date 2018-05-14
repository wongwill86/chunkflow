class InferenceEngine(object):
    def __init__(self, source):
        self.source = source

    def run_inference(self, bounds):
        pass


class IdentityInference(InferenceEngine):
    def __init__(self, factor=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor

    def run_inference(self, bounds):
        # self.source[bounds] *= self.factor
        print('running identity inference on %s' % (bounds,))
