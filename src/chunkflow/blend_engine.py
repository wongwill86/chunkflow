class BlendEngine(object):
    def _process(self, data):
        """
        returns processed data
        """
        raise NotImplemented

    def run_blend(self, data):
        # self.source[bounds] *= self.factor
        # print('************** running identity Blend on %s' % (bounds,))
        return self._process(data)


class IdentityBlend(BlendEngine):
    def __init__(self, factor=1, *args, **kwargs):
        self.factor = factor

    def _process(self, data):
        return data * self.factor


