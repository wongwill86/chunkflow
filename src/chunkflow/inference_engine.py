class InferenceEngine(object):
    def _process(self, data):
        """
        returns processed data
        """
        raise NotImplemented

    def run_inference(self, data):
        # self.source[bounds] *= self.factor
        return self._process(data)

    def inference_stream(slices, index, input_datasource, output_datasource)

        return (
            Observable.just(slices)
            .do_action(lambda x: print('running inference on %s, %s' % (x, index)))
            .map(input_datasource)
            .map(self.run_inference)
            .map(partial(output_datasource.__setitem__, slices))
        )



class IdentityInference(InferenceEngine):
    def __init__(self, factor=1, *args, **kwargs):
        self.factor = factor

    def _process(self, data):
        return data * self.factor


