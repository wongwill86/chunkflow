import traceback
from collections import deque
from datetime import datetime
from rx import Observable, config
from functools import reduce
from threading import current_thread
from chunkblocks.iterators import UnitIterator
from chunkflow.streams import blocking_subscribe
import itertools
import functools
import linecache
import tracemalloc
import psutil
import time
import os
import numpy as np
from rx import Observable


class RunState:
    def __init__(self, timeout=None):
        self.timeout = timeout
        self.in_flight = 0
        self.has_started = False
        self.latch = config['concurrency'].Event()
        self.buffer_queue = deque()
        self.completed = 0

    def mark_begin_flight(self):
        self.in_flight += 1

    def mark_done_flight(self):
        self.in_flight -= 1
        self.completed += 1

    def add_buffer(self, buffered_chunks):
        self.buffer_queue.append(buffered_chunks)

    def get_buffer(self):
        return self.buffer_queue.popleft()

    def wait_for_completion(self):
        self.latch.wait(self.timeout)

class BlockProcessor:
    def __init__(self, block, on_next=None, on_error=None, on_completed=None, timeout=None):
        self.block = block
        self.num_chunks = reduce(lambda x, y: x * y, block.num_chunks)
        self.completed = 0
        self.error = None
        self.on_next = on_next
        self.on_error = on_error
        self.on_completed = on_completed
        self.timeout = timeout

    def process(self, processing_stream, start_slice=None):
        print('Num_chunks %s' % (self.block.num_chunks,))
        if start_slice:
            start = self.block.slices_to_unit_index(start_slice)
        else:
            start = tuple([0] * len(self.block.bounds))

        run_state = RunState(self.timeout)

        def buffer_complete():
            try:
                process_next_buffered()
            except IndexError:
                run_state.latch.set()
                pass

        def buffer_error(error):
            run_state.latch.set()
            raise error


        def process_next_buffered():
            window = run_state.get_buffer()
            (
                Observable.from_(window)
                .flat_map(processing_stream)
                .subscribe(lambda chunk: self._on_next(chunk, run_state), on_error=buffer_error,
                           on_completed=buffer_complete)
            )

        def enqueue_buffer(buffered_chunks):
            run_state.add_buffer(buffered_chunks)
            if not run_state.has_started:
                run_state.has_started = True
                process_next_buffered()

        Observable.from_(self.block.chunk_iterator(start)).buffer_with_count(count=16).subscribe(
            enqueue_buffer, on_error=self._on_error)
        run_state.wait_for_completion()
        self._on_completed()

    def _on_completed(self):
        print('Finished processing', self.num_chunks)
        if self.on_completed is not None:
            self.on_completed()

    def _on_error(self, error):
        print('\n\n\n\n************************************error************************************\n\n')
        self.error = error
        traceback.print_exception(None, error, error.__traceback__)
        if self.on_error is not None:
            self.on_error(error)
        raise error

    def _on_next(self, chunk, run_state):
        run_state.mark_done_flight()
        print('****** %s--%s %s. %s of %s done ' % (datetime.now(), current_thread().name, chunk.unit_index,
                                                    run_state.completed, self.num_chunks))
        if self.on_next is not None:
            self.on_next(chunk)
