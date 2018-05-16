from collections import namedtuple
from functools import partial
import multiprocessing
import unittest
import traceback
from math import floor
import sys
from threading import current_thread

from rx import Observable
from rx import config
from rx.concurrency import ThreadPoolScheduler
from rx.core.blockingobservable import BlockingObservable
from rx.internal import extensionmethod

from chunkflow.iterators import UnitBFSIterator

import numpy as np

# @extensionmethod(BlockingObservable)
# def blocking_subscribe(source, on_next = None, on_error = None, on_completed = None):
#     """
#     only needed when using subscribe_on
#     https://github.com/ReactiveX/RxPY/issues/203#issuecomment-372963230
#     """
#     latch = config['concurrency'].Event()

#     def onNext(src):
#         if on_next:
#             on_next(src)

#     def onError(src):
#         if on_error:
#             on_error(src)
#         latch.set()

#     def onCompleted():
#         if on_completed:
#             on_completed()
#         latch.set()

#     disposable = source.subscribe(onNext, onError, onCompleted)
#     latch.wait()

#     return disposable

class BlockProcessor(object):
    def __init__(self, inference_engine, blend_engine, datasource_manager, block_size, overlap=None,
                 base_iterator=None):
        self.inference_engine = inference_engine
        self.blend_engine = blend_engine
        self.datasource_manager = datasource_manager

        if not overlap:
            overlap = tuple([0] * len(block_size))
        if not base_iterator:
            base_iterator = UnitBFSIterator()

        self.overlap = overlap
        self.block_size = block_size

        self.stride = tuple((b_size - olap) for b_size, olap in zip(self.block_size, self.overlap))
        self.base_iterator = base_iterator

    def _unit_index_to_slices(self, bounds, index):
        return tuple(slice(b.start + idx * s, b.start + idx * s + b_size) for b, idx, s, b_size in zip(
            bounds, index, self.stride, self.block_size))

    def _slices_to_unit_index(self, bounds, slices):
        return tuple(floor((slice.start - b.start) / s) for b, s, slice in zip(bounds, self.stride, slices))

    def _bounds_to_block_sizes(self, bounds):
        data_size = tuple(b.stop - b.start for b in bounds)
        num_blocks = tuple(floor(d_size / s) for d_size, s in zip(data_size, self.stride))
        for blocks, b_size, d_size, olap in zip(num_blocks, self.block_size, data_size, self.overlap):
            if blocks * (b_size - olap) + olap != d_size:
                raise ValueError('Data size %s divided by %s with overlap %s does not divide evenly' % (
                    data_size, self.block_size, self.overlap))
        return  num_blocks, data_size

    def upload(self, combined_index, data=None):
        print('uploading %s data' % (combined_index,))

    def process(self, bounds, start_slice=None):
        num_blocks, data_size = self._bounds_to_block_sizes(bounds)
        optimal_thread_count = multiprocessing.cpu_count()
        scheduler = ThreadPoolScheduler(optimal_thread_count)

        done = set()

        print('num_blocks %s(' % (num_blocks,))
        # TODO refactor
        if start_slice:
            start = self._slices_to_unit_index(start_slice)
        else:
            start = tuple([0] * len(bounds))


        index_to_slices_partial = partial(self._unit_index_to_slices, bounds)
        slices_to_index_partial = partial(self._slices_to_unit_index, bounds)
        to_combined_index = lambda unit_index: CombinedIndex(unit_index=unit_index,
                                                             slices=index_to_slices_partial(unit_index))

        datasource_observable = Observable.from_(self.datasource_manager.repository.values())


        def inference_stream(index):
            slices = index_to_slices_partial(index)
            return (
                Observable.just(slices)
                .do_action(lambda x: print('running inference on %s, %s' % (x, index)))
                .map(self.datasource_manager.input_datasource.__getitem__)
                .map(self.inference_engine.run_inference)
                .map(partial(self.datasource_manager.get_datasource(index).__setitem__, slices))
                .map(lambda data: index)
            )

        def aggregate(slices, aggregate, datasource):
            aggregate += datasource[slices]
            return aggregate

        def blend_stream(index):
            slices = index_to_slices_partial(index)
            return (
                Observable.combine_latest(
                    Observable.just(index).map(self.datasource_manager.get_datasource),
                    datasource_observable.reduce(partial(aggregate, slices), seed=np.zeros(self.block_size)
                    ).map(self.blend_engine.run_blend),
                    lambda output_datasource, data: output_datasource.__setitem__(slices, data)
                )
                .map(lambda data: index)
            )


        ready_index_stream = (
            Observable.from_(self.base_iterator.get(start, num_blocks))
            .flat_map(lambda index:
                      (
                          Observable.just(index).zip(
                              index_to_slices_partial(index),
                              self.datasource_manager.input_datasource,
                              self.datasource_manager.get_datasource(index)
                          )
                          .map(self.inference_engine.inference_stream)
                          .map(lambda data: index)
                      )

            .flat_map(inference_stream)
            .do_action(lambda index: done.add(index))
            .flat_map(lambda index: self.base_iterator.get_all_neighbors(index, max=num_blocks))
            # multithread
            .flat_map(lambda x: Observable.just(x, scheduler=scheduler))
            .filter(
                lambda index:
                all(neighbor in done for neighbor in self.base_iterator.get_all_neighbors(index, max=num_blocks)))
            .distinct()
            .flat_map(blend_stream)
            # .flat_map(upload_stream)
            .subscribe(
                self.upload,
                on_error=lambda error: print('error error *&)*&*&)*\n\n') or traceback.print_exception(
                    None, error, error.__traceback__))
        )

        # def inner_upload(combined_index):
        #     datasource = self.datasource_manager.get_datasource(combined_index.unit_index)
        #     datasource_master_core = self.datasource_manager.get_core()
        #     datasource_master_core[combined_index.slices] = datasource[combined_index.slices]

        # def edge_upload(combined_index):
        #     datasource = self.datasource_manager.get_datasource(combined_index.unit_index)
        #     slices_core = self._unit_index_to_core_slices(bounds, combined_index.unit_indexndex)
        #     slices_edges = self._unit_index_to_edge_slices(bounds, combined_index.unit_indexndex)

        #     datasource_core = self.datasource_manager.get_core()
        #     datasource_edge = self.datasource_manager.get_edge()

        #     datasource_core[slices_core] = datasource[slices_core]

        #     for slices_edge in slices_edges:
        #         datasource_edge[slices_edge] = datasource[slices_edge]

        # split between inner blocks and edge/corner blocks
        # edge_stream, inner_stream = neighbor_stream.partition(
        #     lambda index: any(idx == 0 or idx == max_index for idx, max_index in zip(index.unit_index, num_blocks)))
