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
CombinedIndex = namedtuple('CombinedIndex', 'unit_index slices')
IndexDatasource = namedtuple('IndexDatasource', 'combined_index datasource')

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

    def run_inference(self, combined_index):
        print('---------------runnin inf on %s' % (combined_index,))
        datasource = self.datasource_manager.get_datasource(combined_index.unit_index)
        datasource[combined_index.slices] = self.inference_engine.run_inference(
            self.datasource_manager.input_datasource, combined_index.slices)

    def blend(self, combined_index):
        print('****************runnin blend on %s' % (combined_index,))
        datasource = self.datasource_manager.get_datasource(combined_index.unit_index)
        datasource[combined_index.slices] = self.blend_engine.run_blend(datasource, combined_index.slices)

    def upload(self, combined_index, data=None):
        print('uploading %s data' % (combined_index,))

    def blend_and_upload(self, combined_index):
        def in_place_sum(aggregate, datasource):
            aggregate += datasource[combined_index.slices]
            return aggregate
        (
            Observable.from_(self.datasource_manager.repository.values())
            .reduce(in_place_sum, seed=np.zeros(self.block_size))
            .subscribe(lambda data: self.upload(combined_index, data))
        )

    def process(self, bounds, start_slice=None):
        num_blocks, data_size = self._bounds_to_block_sizes(bounds)
        optimal_thread_count = multiprocessing.cpu_count()
        scheduler = ThreadPoolScheduler(optimal_thread_count)

        done = set()

        print('num_blocks %s(' % (num_blocks,))
        # TODO refactor
        def run_edge_upload(index):
            print('running edge_upload for %s' % (index,))
            start = tuple([0] * len(bounds))
        def run_inner_upload(index):
            print('running innner_upload for %s' % (index,))
            start = tuple([0] * len(bounds))
        def run_clear(index):
            print('running clear for %s current thread is %s' % (index, current_thread().name))
            start = tuple([0] * len(bounds))

        if start_slice:
            start = self._slices_to_unit_index(start_slice)
        else:
            start = tuple([0] * len(bounds))


        index_to_slices_partial = partial(self._unit_index_to_slices, bounds)
        slices_to_index_partial = partial(self._slices_to_unit_index, bounds)
        to_combined_index = lambda unit_index: CombinedIndex(unit_index=unit_index,
                                                             slices=index_to_slices_partial(unit_index))

        ready_index_stream = (
            Observable.from_(self.base_iterator.get(start, num_blocks))
            .map(to_combined_index)
            # save inference to 1 of 27 datasource choices
            .do_action(self.run_inference)
            .do_action(lambda combined_index: done.add(combined_index.unit_index))
            .flat_map(lambda combined_index: self.base_iterator.get_all_neighbors(combined_index.unit_index,
                                                                                  max=num_blocks))
            # multithread
            # .flat_map(lambda x: Observable.just(x, scheduler=scheduler))
            .map(to_combined_index)
            .filter(lambda combined_index: all(neighbor in done for neighbor in self.base_iterator.get_all_neighbors(
                        combined_index.unit_index, max=num_blocks)))
            .distinct()
        )

        datasource_stream = (

        )

        (
            ready_index_stream
            .flat_map(lambda combined_index:
                      (
                          Observable.combine_latest(Observable.just(combined_index),
                                                    Observable.from_(self.datasource_manager.repository.values()),
                                                    lambda index, datasource: IndexDatasource(combined_index=index,
                                                                                              datasource=datasource))
                          .reduce(lambda aggregate, index_datasource:
                                  aggregate + index_datasource.datasource[index_datasource.combined_index.slices],
                                  seed=np.zeros(self.block_size))
                          .map(lambda x: combined_index)
                      )
            )
            # .subscribe(lambda data: self.upload(combined_index, data))
            .subscribe(
                self.upload,
                on_error=lambda error: print('error error *&)*&*&)*\n\n') or traceback.print_exception(None, error,
                                                                                                       error.__traceback__))
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


        # edge_stream = (
        #     edge_stream
        #     .distinct()
        #     # get inference from 1 of 27 datasources and upload appropriate edges to
        #     # single master core and single master edge datasource
        # )

        # inner_stream = (
        #     inner_stream
        #     .filter(lambda combined_index: all(
        #         neighbor in done for neighbor in self.base_iterator.get_all_neighbors(combined_index.unit_index)))
        #     .distinct()
        #     # get inference from all 27 datasource and blend into main at index
        #     .do_action(self.blend)
        #     # upload main at index to master core
        # )

        # Observable.merge(edge_stream, inner_stream).do_action(self.upload).subscribe(
        #     run_clear,
        #     on_error=lambda error: print('error error *&()*&*(&*(&)*(\n\n') or traceback.print_exception(None, error, error.__traceback__))

        # to avoid numpy data copy

        # def blend_and_upload(combined_index):
        #     def in_place_sum(aggregate, datasource):
        #         aggregate += datasource[combined_index.slices]
        #         return aggregate
        #     (
        #         Observable.from_(self.datasource_manager.repository.values())
        #         .scan(in_place_sum, seed=numpy.zeros(self.block_size))
        #         .subscribe(run_clear)
        #     )
        #     print('****************runnin blend on %s' % (combined_index,))
        #     datasource = self.datasource_manager.get_datasource(combined_index.unit_index)
        #     datasource[combined_index.slices] = self.blend_engine.run_blend(datasource, combined_index.slices)

        # ready_index_stream = (
        #     neighbor_stream
        #     .filter(lambda combined_index: all(neighbor in done for neighbor in self.base_iterator.get_all_neighbors(
        #                 combined_index.unit_index, max=num_blocks)))
        #     .distinct()
        # )
        # datasource_stream = (
        #     Observable.from_(self.datasource_manager.repository.values())
        # )
        # def in_place_sum(aggregate, index_datasource):
        #     combined_index, datasource = index_datasource
        #     aggregate += datasource[combined_index.slices]
        #     return aggregate

        # (
        #     Observable.combine_latest(ready_index_stream, datasource_stream,
        #                               lambda combined_index, datasource: (combined_index, datasource))
        #     .scan(in_place_sum, seed=np.zeros(self.block_size))
        #     .map()
        #     .subscribe(
        #         run_clear,
        #         on_error=lambda error: print('error error *&)*&*&)*\n\n') or traceback.print_exception(None, error,
        #                                                                                                error.__traceback__))

        # )
