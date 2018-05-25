import traceback
from datetime import datetime
from functools import partial
from threading import current_thread

import numpy as np
from rx import Observable
from rx import config
# from rx.concurrency import ThreadPoolScheduler
from rx.core.blockingobservable import BlockingObservable
from rx.internal import extensionmethod


@extensionmethod(BlockingObservable)
def blocking_subscribe(source, on_next=None, on_error=None, on_completed=None):
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
    def __init__(self, inference_operation, blend_operation, datasource_manager):
        self.inference_operation = inference_operation
        self.blend_operation = blend_operation
        self.datasource_manager = datasource_manager
        self.datasource_stream = Observable.from_(self.datasource_manager.repository.repository.values())

    def process(self, block, start_slice=None):
        # optimal_thread_count = multiprocessing.cpu_count()
        # scheduler = ThreadPoolScheduler(optimal_thread_count)

        print('num_chunks %s' % (block.num_chunks,))
        if start_slice:
            start = self.block.slices_to_unit_index(start_slice)
        else:
            start = tuple([0] * len(block.bounds))

        (
            Observable.from_(block.chunk_iterator(start))
            # .do_action(lambda chunk: chunk.load_data(self.datasource_manager.input_datasource))
            .do_action(self.datasource_manager.load_chunk)
            .map(self.inference_operation)
            .map(self.blend_operation)
            # .flat_map(
            #     lambda chunk:
            #     (
            #         Observable.just(chunk.data)
            #         .do_action(lambda x: print('>>>>>> %s--%s %s running inference' % (
            #                         datetime.now(), current_thread().name, chunk.unit_index)))
            #         .map(self.inference_operation.run_inference)
            #         .map(self.blend_operation.run_blend)
            #         .map(chunk.load_data)
            #         .map(chunk)
            #     )
            # )
            .do_action(self.datasource_manager.dump_chunk)
            # .do_action(lambda chunk: chunk.dump_data(self.datasource_manager.get_datasource(chunk.unit_index)))
            .do_action(block.checkpoint)
            .flat_map(block.get_all_neighbors)
            .filter(block.all_neighbors_checkpointed)
            .distinct()
            # put on new thread
            # .flat_map(lambda neighbor_index: Observable.from_((neighbor_index,), scheduler=scheduler))
            # sum the different datasources together
            .flat_map(
                lambda chunk:
                (
                    self.datasource_stream
                    .reduce(partial(aggregate, chunk.slices), seed=np.zeros(block.chunk_size))
                    .do_action(chunk.load_data)
                    .map(lambda _: chunk)
                )
            )
            .flat_map(lambda chunk:
                      Observable.merge(
                          Observable.just(chunk).flat_map(block.overlap_slices).do_action(
                              partial(self.datasource_manager.upload_overlap, chunk)),
                          Observable.just(chunk).map(block.core_slices).do_action(
                              partial(self.datasource_manager.upload_core, chunk))
                      )
                      )
            # .do_action(lambda chunk: chunk.dump_data(self.datasource_manager.get_datasource(chunk.unit_index)))
            .subscribe(
                #self.datasource_manager.upload(sub_chunk)
                print,
                on_error=lambda error: print('error error *&)*&*&)*\n\n') or traceback.print_exception(
                    None, error, error.__traceback__))
        )

    def upload_chunk(self, chunk, data=None):
        print('****** %s--%s %s uploading data' % (datetime.now(), current_thread().name, chunk.unit_index,))
