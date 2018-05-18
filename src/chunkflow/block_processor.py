from collections import namedtuple
from functools import partial
import multiprocessing
import unittest
import traceback
from math import floor
import sys
from threading import current_thread
from  datetime import datetime

from rx import Observable
from rx import config
from rx.core import Scheduler
from rx.concurrency import ThreadPoolScheduler
from rx.core.blockingobservable import BlockingObservable
from rx.internal import extensionmethod

from chunkflow.iterators import UnitBFSIterator

import numpy as np

@extensionmethod(BlockingObservable)
def blocking_subscribe(source, on_next = None, on_error = None, on_completed = None):
    """
    only needed when using subscribe_on
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

def aggregate(slices, aggregate, datasource):
    aggregate += datasource[slices]
    return aggregate

class BlockProcessor(object):
    def __init__(self, inference_engine, blend_engine, datasource_manager):
        self.inference_engine = inference_engine
        self.blend_engine = blend_engine
        self.datasource_manager = datasource_manager
        self.datasource_stream = Observable.from_(self.datasource_manager.repository.values())

    # def run_inference(chunk):
    #     return (
    #         Observable.just(chunk.data)
    #     .do_action(lambda x: print('running inference on %s, %s' % (chunk.unit_idex, chunk.slices)))
    #     .map(self.inference_engine.run_inference)
    #     .do_action(chunk.set_data)
    #     .map(lambda _: chunk)

    def process(self, block, start_slice=None):
        optimal_thread_count = multiprocessing.cpu_count()
        scheduler = ThreadPoolScheduler(optimal_thread_count)
        single = ThreadPoolScheduler(1)

        done = set()

        print('num_chunks %s' % (block.num_chunks,))
        if start_slice:
            start = self.block._slices_to_unit_index(start_slice)
        else:
            start = tuple([0] * len(block.bounds))

        def run_inference(chunk):
            return (
                Observable.just(chunk.data)
                .do_action(lambda x: print('>>>>>> %s--%s %s running inference' % (datetime.now(), current_thread().name, chunk.unit_index)))
                .map(self.inference_engine.run_inference)
                .map(self.blend_engine.run_blend)
                .map(lambda _: chunk)
            )
        source = Observable.from_(block.get_iterator(start))

        unit_index_to_chunk = partial(Chunk, block)

        stream = (
            # Observable.from_(block.get_iterator(start))
            source
            .map(unit_index_to_chunk)
            .do_action(lambda chunk: chunk.load_data(self.datasource_manager.input_datasource))
            .flat_map(run_inference)
            .do_action(lambda chunk: chunk.dump_data(self.datasource_manager.get_datasource(chunk.unit_index)))
            .map(lambda chunk: chunk.unit_index)
            .do_action(block.checkpoint)
            .flat_map(block.get_all_neighbors)
            .filter(block.all_neighbors_checkpointed)
            .distinct()
            # put on new thread
            # .flat_map(lambda neighbor_index: Observable.from_((neighbor_index,), scheduler=scheduler))
            .map(unit_index_to_chunk)
            # .do_action(lambda chunk: chunk.load_data(self.datasource_manager.input_datasource))
            .flat_map(
                lambda chunk:
                (
                    self.datasource_stream
                    .reduce(partial(aggregate, chunk.slices), seed=np.zeros(block.chunk_size))
                    .do_action(chunk.load_data)
                    .map(lambda _: chunk)
                )
            )
            .do_action(lambda chunk: chunk.dump_data(self.datasource_manager.get_datasource(chunk.unit_index)))
            .subscribe(
                self.upload,
                on_error=lambda error: print('error error *&)*&*&)*\n\n') or traceback.print_exception(
                    None, error, error.__traceback__))
        )

    def upload(self, chunk, data=None):
        print('****** %s--%s %s uploading data' % (datetime.now(), current_thread().name, chunk.unit_index,))


class Chunk(object):
    def __init__(self, block, unit_index):
        self.unit_index = unit_index
        self.slices = block.unit_index_to_slices(unit_index)
        self.data = None
        self.size = block.chunk_size

    def load_data(self, datasource):
        import time
        print('VVVVVV %s--%s %s loading into chunk' % (datetime.now(), current_thread().name, self.unit_index))
        self.data = datasource[self.slices]

    def dump_data(self, datasource):

        print('^^^^^^ %s--%s %s dumping from chunk' % (datetime.now(), current_thread().name, self.unit_index))
        import time
        # time.sleep(.1)
        datasource[self.slices] = self.data

    def get_core_slices(self):
        pass

    def get_edge_slices(self):
        pass


class Block(object):
    def __init__(self, bounds, chunk_size, overlap=None, base_iterator=None):
        self.bounds = bounds
        self.chunk_size = chunk_size

        if not overlap:
            overlap = tuple([0] * len(chunk_size))

        self.overlap = overlap
        if not base_iterator:
            base_iterator = UnitBFSIterator()
        self.base_iterator = base_iterator


        self.stride = tuple((b_size - olap) for b_size, olap in zip(self.chunk_size, self.overlap))
        self.num_chunks = self.calc_num_chunks(bounds)
        self.checkpoints = set()

    def unit_index_to_slices(self, index):
        return tuple(slice(b.start + idx * s, b.start + idx * s + b_size) for b, idx, s, b_size in zip(
            self.bounds, index, self.stride, self.chunk_size))

    def slices_to_unit_index(self, bounds, slices):
        return tuple(floor((slice.start - b.start) / s) for b, s, slice in zip(bounds, self.stride, slices))

    # TODO
    def calc_num_chunks(self, bounds):
        data_size = tuple(b.stop - b.start for b in bounds)
        num_chunks = tuple(floor(d_size / s) for d_size, s in zip(data_size, self.stride))
        for blocks, b_size, d_size, olap in zip(num_chunks, self.chunk_size, data_size, self.overlap):
            if blocks * (b_size - olap) + olap != d_size:
                raise ValueError('Data size %s divided by %s with overlap %s does not divide evenly' % (
                    data_size, self.chunk_size, self.overlap))
        return  num_chunks

    def checkpoint(self, index):
        self.checkpoints.add(index)

    def checkpoint_complete(self, indices):
        return all(index in checkpoints for index in indices)

    def get_all_neighbors(self, index):
        return self.base_iterator.get_all_neighbors(index, max=self.num_chunks)

    def all_neighbors_checkpointed(self, index):
        return all(neighbor in self.checkpoints for neighbor in self.get_all_neighbors(index))

    def get_iterator(self, start):
        yield from self.base_iterator.get(start, self.num_chunks)


