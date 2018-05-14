import multiprocessing
import unittest
from math import floor
from threading import current_thread

from rx import Observable
from rx import config
from rx.concurrency import ThreadPoolScheduler
from rx.core.blockingobservable import BlockingObservable
from rx.internal import extensionmethod

from chunkflow.iterators import UnitBFSIterator


@extensionmethod(BlockingObservable)
def blocking_subscribe(source, on_next = None, on_error = None, on_completed = None):
    """
    https://github.com/ReactiveX/RxPY/issues/203#issuecomment-372963230
    """
    latch = config['concurrency'].Event()

    def onNext(src):
        if on_next:
            on_next(src)

    def onError(src):
        if on_error:
            on_error(src)
        latch.set()

    def onCompleted():
        if on_completed:
            on_completed()
        latch.set()

    disposable = source.subscribe(onNext, onError, onCompleted)
    latch.wait()

    return disposable

class BlockProcessor(object):
    def __init__(self, inference_engine, blend_engine, block_size, overlap=None, base_iterator=UnitBFSIterator()):
        self.inference_engine = inference_engine
        self.blend_engine = blend_engine

        if not overlap:
            overlap = tuple([0] * len(block_size))

        self.overlap = overlap
        self.block_size = block_size

        self.stride = tuple((b_size - olap) for b_size, olap in zip(self.block_size, self.overlap))
        self.base_iterator = base_iterator

    def _unit_index_to_slices(self, bounds, index):
        return tuple(slice(b.start + idx * s, b.start + idx * s + b_size) for b, idx, s, b_size in zip(
            bounds, index, self.stride, self.block_size))

    def _slices_to_unit_index(self, bounds, slices):
        return NotImplemented

    def _bounds_to_block_sizes(self, bounds):
        data_size = tuple(b.stop - b.start for b in bounds)
        num_blocks = tuple(floor(d_size / s) for d_size, s in zip(data_size, self.stride))
        for blocks, b_size, d_size, olap in zip(num_blocks, self.block_size, data_size, self.overlap):
            if blocks * (b_size - olap) + olap != d_size:
                raise ValueError('Data size %s divided by %s with overlap %s does not divide evenly' % (
                    data_size, self.block_size, self.overlap))
        return  num_blocks, data_size

    def process(self, bounds, start_slice=None):
        num_blocks, data_size = self._bounds_to_block_sizes(bounds)
        optimal_thread_count = multiprocessing.cpu_count()
        scheduler = ThreadPoolScheduler(optimal_thread_count)

        print('num blocks %s' % (num_blocks,))
        # bounds = (slice(0, 70), slice(0, 70))
        # block_size = (30, 30)
        # overlap = (10, 10)
        done = set()

        def all_neighbors_done(index):
            return all(neighbor in done for neighbor in self.base_iterator.get_all_neighbor_slices(self.overlap, index))

        def is_bounds_edge(index):
            return any(idx == 0 or idx == max_index for idx, max_index in zip(index, num_blocks))

        def iterate(num_blocks):
            for index in self.base_iterator(start, num_blocks):
                yield block

        # TODO refactor
        def run_edge_upload(index):
            print('running edge_upload for %s' % (index,))
        def run_inner_upload(index):
            print('running innner_upload for %s' % (index,))
        def run_clear(index):
            print('running clear for %s' % (index,))
            start = tuple([0] * len(bounds))

        if start_slice:
            start = self._slices_to_unit_index(start_slice)
        else:
            start = tuple([0] * len(bounds))

        neighbor_stream = (
            Observable.from_(self.base_iterator.iterator(start, num_blocks))
            .do_action(lambda x: print(x))
            .do_action(lambda unit_index:
                       self.inference_engine.run_inference(self._unit_index_to_slices(bounds, unit_index)))
            .do_action(lambda x: done.add(x))
            .flat_map(self.base_iterator.get_all_neighbors)
            .flat_map(lambda x: Observable.just(x, scheduler=scheduler))
        )

        edge_stream, inner_stream = neighbor_stream.partition(is_bounds_edge)

        edge_stream = (
            edge_stream.distinct().do_action(run_edge_upload)
        )

        inner_stream = (
            inner_stream
            .filter(all_neighbors_done)
            .distinct()
            .do_action(self.blend_engine.run_blend)
            .do_action(run_inner_upload)
        )

        # edge_stream.subscribe(run_clear)
        # inner_stream.subscribe(run_clear)
        Observable.merge(edge_stream, inner_stream).subscribe(run_clear); #.to_blocking().blocking_subscribe(run_clear)
        import time
        time.sleep(3)

