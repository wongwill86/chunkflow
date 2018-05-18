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
    def __init__(self, inference_engine, blend_engine, datasource_manager):
        self.inference_engine = inference_engine
        self.blend_engine = blend_engine
        self.datasource_manager = datasource_manager
        self.datasource_stream = Observable.from_(self.datasource_manager.repository.values())

    def process(self, block, start_slice=None):
        # optimal_thread_count = multiprocessing.cpu_count()
        # scheduler = ThreadPoolScheduler(optimal_thread_count)

        print('num_chunks %s' % (block.num_chunks,))
        if start_slice:
            start = self.block._slices_to_unit_index(start_slice)
        else:
            start = tuple([0] * len(block.bounds))

        def run_inference(chunk):
            return (
            )

        (
            Observable.from_(block.chunk_iterator(start))
            .do_action(lambda chunk: chunk.load_data(self.datasource_manager.input_datasource))
            .flat_map(
                lambda chunk:
                (
                    Observable.just(chunk.data)
                    .do_action(lambda x: print('>>>>>> %s--%s %s running inference' % (
                                    datetime.now(), current_thread().name, chunk.unit_index)))
                    .map(self.inference_engine.run_inference)
                    .map(self.blend_engine.run_blend)
                    .map(lambda _: chunk)
                )
            )
            .do_action(lambda chunk: chunk.dump_data(self.datasource_manager.get_datasource(chunk.unit_index)))
            .do_action(block.checkpoint)
            .flat_map(block.get_all_neighbors)
            .filter(block.all_neighbors_checkpointed)
            .distinct()
            # put on new thread
            # .flat_map(lambda neighbor_index: Observable.from_((neighbor_index,), scheduler=scheduler))
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
