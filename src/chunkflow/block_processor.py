import traceback
from datetime import datetime
from functools import partial
from threading import current_thread

from rx import Observable


class BlockProcessor:
    def process(self, block, processing_stream, start_slice=None):
        print('num_chunks %s' % (block.num_chunks,))
        if start_slice:
            start = self.block.slices_to_unit_index(start_slice)
        else:
            start = tuple([0] * len(block.bounds))

        # for chunk in block.chunk_iterator(start):
        #     processing_stream.on_next(chunk)

        (
            Observable.from_(block.chunk_iterator(start))
            .flat_map(processing_stream)
            .to_blocking()
            .blocking_subscribe(self.print_done, on_error=self.on_error)
        )

    def on_error(self, error):
        print('\n\n\n\nerror error *&)*&*&)*\n\n')
        traceback.print_exception(None, error, error.__traceback__)
        raise error

    def print_done(self, chunk, data=None):
        print('****** %s--%s %s done ' % (datetime.now(), current_thread().name, chunk.unit_index,))
