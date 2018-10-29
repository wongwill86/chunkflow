from collections import namedtuple
from enum import Enum
from functools import partial

import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from chunkblocks.models import Chunk
from rx import Observable, config
from rx.core import AnonymousObservable
from rx.core.blockingobservable import BlockingObservable
from rx.internal import extensionmethod
from memory_profiler import profile
import gc

from chunkflow.chunk_buffer import CacheMiss

MAX_RETRIES = 10


def del_data(chunk):
    if hasattr(chunk, 'data'):
        # objgraph.show_backrefs([chunk.data], filename='futs/data%s.png' % id(chunk.data))
        # print('showing omost common types for ', id(chunk.data))
        # objgraph.show_most_common_types(objects=[chunk.data])
        del chunk.data
        chunk.data = None

from rx.core import Observable, AnonymousObservable
from rx.internal.utils import is_future
from rx.internal import extensionclassmethod
import objgraph


@extensionclassmethod(Observable)
def from_my_future(cls, future):
    """Converts a Future to an Observable sequence
    Keyword Arguments:
    future -- {Future} A Python 3 compatible future.
        https://docs.python.org/3/library/asyncio-task.html#future
        http://www.tornadoweb.org/en/stable/concurrent.html#tornado.concurrent.Future
    Returns {Observable} An Observable sequence which wraps the existing
    future success and failure.
    """

    def subscribe(observer):
        def done(future):
            try:
                value = future.result()
                # if hasattr(value, 'data') and value.data is not None:
                #     chunk_copy = value.block.unit_index_to_chunk(value.unit_index)
                #     chunk_copy.data = value.data.copy()
                #     # objgraph.show_backrefs([blah], filename='futs/fut%s.png' % id(blah))
                #     # print('showing omost common types for ', id(blah))
                #     # objgraph.show_most_common_types(objects=[blah])
                #     if hasattr(value, 'data'):
                #         del value.data
                #     del value
                #     del future
                #     value = chunk_copy


                # objgraph.show_backrefs([future], filename='futs/future%s.png' % id(future))
                # print('showing omost common types for ', id(future))
                # objgraph.show_most_common_types(objects=[future])
                # del future
            except Exception as ex:
                observer.on_error(ex)
            else:
                observer.on_next(value)
                observer.on_completed()

        future.add_done_callback(done)

        def dispose():
            if future and future.cancel:
                future.cancel()

        return dispose

    return AnonymousObservable(subscribe) if is_future(future) else future

from multiprocessing import Pool
pool = dict()  # Pool(maxtasksperchild=20)

@extensionmethod(Observable)
def from_item_or_future(item_or_future, default=None):
    """
    Checks the item to see if it is a future-like. (could be asyncio future which is not part of the concurrent.futures
    package).
    """

    def exec_subscribe(observer):
        def done(value):
            print('\n\n\n\n\n I AM DONE')
            observer.on_next(value)
            observer.on_completed()

        result = pool.apply_async(item_or_future[0], item_or_future[1:], callback=done, error_callback=observer.on_error)

        def dispose():
            print('ugh trying to cancel blah thi sis not good')
            del result
            pass
            if future and future.cancel:
                future.cancel()

        return dispose


    if isinstance(item_or_future, tuple):
        return AnonymousObservable(exec_subscribe)

    if hasattr(item_or_future, 'result'):  # should probably check if it is callable too
        return Observable.from_my_future(item_or_future)
    elif item_or_future is None:
        return Observable.of(default)
    else:
        try:
            iter(item_or_future)
        except TypeError:
            return Observable.of(item_or_future)
        else:
            return Observable.from_(item_or_future).flat_map(from_item_or_future)


@extensionmethod(Observable)
def distinct_hash(self, key_selector=None, seed=None, lock=None):
    """
    Returns an observable sequence that contains only distinct elements according to the key_mapper. Usage of
    this operator should be considered carefully due to the maintenance of an internal lookup structure which can grow
    large. This implementation allows specifying a given set to track duplicate keys.
    Example:
    res = obs = xs.distinct_hash()
    obs = xs.distinct_hash(key_selector=lambda x: x.id)
    obs = xs.distinct_hash(key_selector=lambda x: x.id, lambda a,b: a == b)
    Keyword arguments:
    key_selector -- [Optional]  A function to compute the comparison key
        for each element.
    seed -- [Optional]  Set used to track keys with. Use to enable global tracking.
    Returns an observable sequence only containing the distinct
    elements, based on a computed key value, from the source sequence.
    """
    source = self
    hashset = seed if seed is not None else set()

    def subscribe(observer, scheduler=None):

        def on_next(x):
            key = x
            if key_selector:
                try:
                    key = key_selector(x)
                except Exception as ex:
                    observer.on_error(ex)
                    return

            def add_and_next():
                if key not in hashset:
                    hashset.add(key)
                    observer.on_next(x)

            if lock is None:
                add_and_next()
            else:
                with lock:
                    add_and_next()

        return source.subscribe(on_next, observer.on_error, observer.on_completed, scheduler)
    return AnonymousObservable(subscribe)


@extensionmethod(BlockingObservable)
def blocking_subscribe(source, on_next=None, on_error=None, on_completed=None, timeout=None):
    """
    only needed when using subscribe_on
    https://github.com/ReactiveX/RxPY/issues/203#issuecomment-372963230
    """
    latch = config['concurrency'].Event()

    def onNext(src):
        if on_next:
            on_next(src)

    def onError(src):
        try:
            if on_error:
                on_error(src)
            else:
                raise src
        finally:
            latch.set()

    def onCompleted():
        try:
            if on_completed:
                on_completed()
        finally:
            latch.set()

    disposable = source.subscribe(onNext, onError, onCompleted)
    latch.wait(timeout)

    return disposable


# @profile
def aggregate(slices, aggregate, datasource):
    try:
        # Account for additional output dimensions
        channel_dimensions = len(datasource.shape) - len(slices)
        channel_slices = (slice(None),) * (channel_dimensions) + slices
    except ReferenceError:
        return aggregate

    try:
        data = datasource[channel_slices]
    except CacheMiss:
        data = datasource.get_item(channel_slices, fill_missing=True)

    # 0 from seed
    if aggregate is 0:
        slice_shape = tuple(s.stop - s.start for s in slices)
        offset = (0,) * channel_dimensions + tuple(s.start for s in slices)

        data = GlobalOffsetArray(
            np.zeros(data.shape[0:channel_dimensions] + slice_shape, dtype=data.dtype),
            global_offset=offset
        )
        # aggregate_chunk.load_data(GlobalOffsetArray(
        #     np.zeros(data.shape[0:channel_dimensions] + slice_shape, dtype=data.dtype),
        #     global_offset=offset
        # ))
        # aggregate_chunk.load_data(aggregate_chunk)
        aggregate = data
    else:
        aggregate[channel_slices] += data

    return aggregate


def download_retry(datasource_manager, datasource, patch_chunk):
    """
    Try to load from cache if available
    returns either:
        Future - flatmap
        Observable - flatmap
    """
    buffered_datasource = datasource_manager.get_buffer(datasource)
    use_executor = buffered_datasource is None

    def add_future_to_cache(datasource_chunk):  # if multithreading, this needs to be synchronized
        if datasource_chunk.unit_index not in buffered_datasource.local_cache:
            future = datasource_manager.load_chunk(
                datasource_chunk, datasource=datasource, use_buffer=False)
            buffered_datasource.local_cache[datasource_chunk.unit_index] = future
        return Observable.from_item_or_future(buffered_datasource.local_cache[datasource_chunk.unit_index])

    def save_to_cache(datasource_chunk):
        if not hasattr(buffered_datasource.local_cache[datasource_chunk.unit_index], 'data'):
            datasource_manager.dump_chunk(datasource_chunk, datasource=datasource, use_executor=False)
        return datasource_chunk

    try:
        # try loading chunk and put in into an observable for consistency
        return Observable.from_item_or_future(datasource_manager.load_chunk(
            patch_chunk, datasource=datasource, use_executor=use_executor))
    except CacheMiss as cm:
        return (
            Observable.from_(cm.misses)
            # creates a temp chunk to throw away
            .map(buffered_datasource.block.unit_index_to_chunk)
            .flat_map(add_future_to_cache)
            .do_action(save_to_cache)
            .reduce(lambda x, y: patch_chunk)
            .map(lambda _: datasource_manager.load_chunk(patch_chunk, datasource=datasource, use_executor=False))
        )


def create_input_stream(datasource_manager):
    return lambda patch_chunk: (
        Observable.just(patch_chunk)
        .flat_map(partial(download_retry, datasource_manager, datasource_manager.input_datasource))
    )


def create_inference_stream(block, inference_operation, blend_operation, datasource_manager):
    return lambda chunk: (
        Observable.just(chunk)
        .do_action(
            lambda chunk:
            (print('before inference', chunk.unit_index) and False) or (datasource_manager.print_cache_stats() and False)
        )
        .map(inference_operation)
        .do_action(
            lambda chunk:
            (print('before blend', chunk.unit_index) and False) or (datasource_manager.print_cache_stats() and False)
        )
        .map(blend_operation)
        .do_action(
            lambda chunk:
            (print('before dump', chunk.unit_index) and False) or (datasource_manager.print_cache_stats() and False)
        )
        .do_action(lambda chunk: datasource_manager.dump_chunk(
            chunk, datasource=datasource_manager.overlap_repository.get_datasource(chunk.unit_index), use_executor=False
        ))
        .do_action(
            lambda chunk:
            (objgraph.show_backrefs([chunk.data], filename='futs/infdata%s-%s-%s.png' % (chunk.unit_index)) and False) or
            (print('showing omost common types for ', chunk.unit_index) and False) or
            (objgraph.show_most_common_types(objects=[chunk.data]) and False)
        )
        .do_action(del_data)
    )


def create_aggregate_stream(block, datasource_manager):
    return lambda chunk: (
        # sum the different datasources together
        Observable.just(chunk)
        .do_action(
            lambda chunk:
            (print('before aggregate', chunk.unit_index) and False) or (datasource_manager.print_cache_stats() and False)
        )
        .flat_map(
            lambda chunk:
            (
                # create temp list of repositories values at time of iteration / also weakref
                Observable.from_(list(datasource_manager.overlap_repositories()))
                .reduce(partial(aggregate, chunk.slices), seed=0)
                .do_action(chunk.load_data)
                .map(lambda _: chunk)
                .do_action(del_data)
            )
        )
        .do_action(
            lambda chunk:
            (print('after aggregate', chunk.unit_index) and False) or (datasource_manager.print_cache_stats() and False)
        )
    )


DumpArguments = namedtuple('DumpArguments', 'datasource slices use_executor')


def create_upload_stream(block, datasource_manager):
    output_datasource_final_buffer = datasource_manager.get_buffer(datasource_manager.output_datasource_final)
    output_datasource_buffer = datasource_manager.get_buffer(datasource_manager.output_datasource)

    use_executor_final = output_datasource_final_buffer is None
    use_executor = output_datasource_buffer is None

    return lambda chunk: (
        Observable.merge(
            # core slices can bypass to the final datasource
            Observable.just(chunk).map(block.core_slices).map(
                lambda slices: DumpArguments(datasource_manager.output_datasource_final, slices, use_executor_final)
            ),
            Observable.just(chunk).flat_map(block.overlap_slices).map(
                lambda slices: DumpArguments(datasource_manager.output_datasource, slices, use_executor)
            )
        )
        .map(lambda dump_args: datasource_manager.dump_chunk(chunk, **dump_args._asdict()))
        .flat_map(Observable.from_item_or_future)
        .reduce(lambda x, y: chunk, seed=chunk).map(lambda _: chunk)  # reduce to wait for all to completed transferring
        .retry(MAX_RETRIES)
        .do_action(del_data)
    )


def create_checkpoint_observable(block, stage, notify_neighbors=True):
    def notify_neighbor_stream(chunk):
        if notify_neighbors:
            return (
                Observable.just(chunk)
                # check both the current chunk we just ran this stage on as well as the neighboring chunks
                .flat_map(lambda chunk: Observable.from_(block.get_all_neighbors(chunk)).start_with(chunk))
                # .do_action(lambda c: print(c.unit_index, 'got to notified by  neighbor 1', chunk.unit_index, 'at',
                #                            stage))
                .filter(lambda chunk: block.is_checkpointed(chunk, stage=stage.value))
                # .do_action(lambda c: print(
                #     c.unit_index, 'got to notified by  neighbor 2', chunk.unit_index, 'at',
                #     stage, 'looking at neighbors',
                #     list(map(lambda x: x.unit_index, block.get_all_neighbors(c))),
                #     'is al lcheckpoined?',
                #     block.all_neighbors_checkpointed(chunk, stage=stage.value)
                #     )
                # )
                .filter(lambda chunk: block.all_neighbors_checkpointed(chunk, stage=stage.value))
                # .do_action(lambda c: print(c.unit_index, 'got to notified by  neighbor 3', chunk.unit_index, 'at',
                #                            stage))
            )
        else:
            return Observable.just(chunk)

    return lambda chunk: (
        Observable.just(chunk)
        .do_action(lambda chunk: block.checkpoint(chunk, stage=stage.value))
        .do_action(lambda chunk: stage.mark_done(chunk.unit_index) or print('Checkpointed:', chunk.unit_index, stage.mark_value, stage, '\nstate:\n', stage.state))
        .flat_map(notify_neighbor_stream)
        .distinct_hash(key_selector=lambda c: c.unit_index, seed=stage.hashset)
        .do_action(lambda chunk: print(chunk.unit_index, 'ready for next', stage))
    )


def create_flush_datasource_observable(datasource_manager, block, stage_to_check, stage_to_complete):
    output_buffer = datasource_manager.get_buffer(datasource_manager.output_datasource)
    output_final_buffer = datasource_manager.get_buffer(datasource_manager.output_datasource_final)

    # assuming buffer blocks are same for both output and output_final
    reference_buffer = (
        output_buffer if output_buffer is not None else
        output_final_buffer if output_final_buffer is not None else None
    )

    if reference_buffer is not None:
        return lambda uploaded_chunk: (
            Observable.from_(reference_buffer.block.slices_to_chunks(uploaded_chunk.slices))
            # .do_action(lambda ds_c: print('looking at datasource chunk at', ds_c.unit_index, 'referred by',
            #     uploaded_chunk.unit_index, 'for', stage_to_check, 'looking at data chunks', stage_to_check.mark_value,
            #     ' at indices', list(map(lambda x: x.unit_index, block.slices_to_chunks(ds_c.slices))),
            #     block.all_checkpointed(block.slices_to_chunks(ds_c.slices), stage=stage_to_check.value)
            # ))
            .filter(lambda datasource_chunk: block.all_checkpointed(
                block.slices_to_chunks(datasource_chunk.slices), stage=stage_to_check.value))
            .do_action(lambda x: print(x.unit_index, 'GREAT SUCCESSS'))
            .flat_map(
                lambda datasource_chunk:
                Observable.from_([datasource_manager.output_datasource, datasource_manager.output_datasource_final])
                # will skip chunks not found i.e. if the cache is already flushed
                .map(partial(datasource_manager.flush, datasource_chunk))
                .flat_map(Observable.from_item_or_future)
                .do_action(del_data)
            )
            # .retry(MAX_RETRIES)
            .do_action(del_data)
            .flat_map(lambda datasource_chunk: block.slices_to_chunks(datasource_chunk.slices))
            .flat_map(create_checkpoint_observable(block, stage_to_complete, notify_neighbors=False))
        )
    else:
        return lambda uploaded_chunk: Observable.just(uploaded_chunk).do_action(lambda x: print('\n\n\n\nUGHGHGHGH'))

def create_inference_stages(block):
    state = np.zeros(block.num_chunks, dtype=np.uint8)
    class Stages(Enum):
        START, INFERENCE_DONE, UPLOAD_DONE, DATASOURCE_FLUSH_DONE, CHUNK_FLUSH_DONE, CLEAR_DONE = range(6)

        def __init__(self, *args, **kwargs):
            self.hashset = set()

        def mark_done(self, index):
            state[index] |= self.mark_value

        @property
        def mark_value(self):
            return 1 << self.value


        @property
        def state(self):
            return state

    return Stages

def create_inference_and_blend_stream(block, inference_operation, blend_operation, datasource_manager):
    stages = create_inference_stages(block)

    return lambda chunk: (
        Observable.just(chunk)
        .flat_map(create_checkpoint_observable(block, stages.START, notify_neighbors=False))
        .do_action(lambda chunk: print('start', chunk.data.nbytes if chunk.data is not None else 0) or
                   datasource_manager.print_cache_stats() or objgraph.show_growth())
        .flat_map(create_input_stream(datasource_manager))
        .flat_map(create_inference_stream(block, inference_operation, blend_operation, datasource_manager))
        .do_action(lambda chunk: print('data after inference', chunk.data.nbytes if chunk.data is not None else 0) or
                   datasource_manager.print_cache_stats() or objgraph.show_growth())
        .flat_map(create_checkpoint_observable(block, stages.INFERENCE_DONE))

        .flat_map(create_aggregate_stream(block, datasource_manager))
        .do_action(lambda chunk: print('data after aggregate', chunk.data.nbytes if chunk.data is not None else 0) or
                   datasource_manager.print_cache_stats() or objgraph.show_growth())
        .flat_map(create_upload_stream(block, datasource_manager))
        .do_action(lambda chunk: print('data after upload', chunk.data.nbytes if chunk.data is not None else 0) or
                   datasource_manager.print_cache_stats() or objgraph.show_growth())
        .flat_map(create_checkpoint_observable(block, stages.UPLOAD_DONE, notify_neighbors=False))
        .do_action(lambda x: print('about to clear buffer input', x.unit_index) or
                   datasource_manager.print_cache_stats() or objgraph.show_growth())
        .do_action(partial(datasource_manager.clear_buffer, datasource_manager.input_datasource))
        .do_action(lambda x: print('ifinsih clear fluhin about flush', x.unit_index) or
                   datasource_manager.print_cache_stats() or objgraph.show_growth())
        .flat_map(create_flush_datasource_observable(datasource_manager, block, stages.UPLOAD_DONE,
                                                     stages.DATASOURCE_FLUSH_DONE))
        # .do_action(lambda chunk: (gc.collect() and False) or print('data after clear', chunk.data.nbytes if chunk.data is not None else 0) or
        #            datasource_manager.print_cache_stats() or objgraph.show_growth())

        .flat_map(create_checkpoint_observable(block, stages.CHUNK_FLUSH_DONE, notify_neighbors=False))
        .do_action(lambda chunk:
                   (datasource_manager.print_cache_stats() and False) or
                   (datasource_manager.overlap_repository.clear(chunk.unit_index) and False) or
                   (print('after clear', chunk.unit_index) and False) or
                   (gc.collect() and False) or (datasource_manager.print_cache_stats() and False)
                   )

        .flat_map(create_checkpoint_observable(block, stages.CLEAR_DONE, notify_neighbors=False))
        .map(lambda _: chunk)
    )


def create_preload_datasource_stream(dataset_block, datasource_manager, datasource):
    output_buffer = datasource_manager.get_buffer(datasource_manager.output_datasource)

    if output_buffer is None:
        raise NotImplementedError("Datasource does not have a buffer therefore, does not require preloading")

    return lambda dataset_chunk: (
        Observable.just(dataset_chunk)
        .flat_map(dataset_block.overlap_chunk_slices)
        .flat_map(output_buffer.block.slices_to_chunks)
        .do_action(lambda datasource_chunk:
                   datasource_manager.copy(datasource_chunk, datasource, destination=datasource))
        .reduce(lambda x, y: dataset_chunk, seed=dataset_chunk)
        .map(lambda _: dataset_chunk)
    )


def create_blend_stream(block, datasource_manager):
    """
    Assume block is a dataset with chunks to represent each task!
    """
    return lambda dataset_chunk: (
        Observable.just(dataset_chunk)
        .flat_map(block.overlap_chunk_slices)
        .flat_map(
            # Aggregate overlap dataset
            lambda dataset_chunk_slices:
            (
                # create temp list of repositories values at time of iteration
                Observable.from_(list(datasource_manager.overlap_repository.datasources.values()))
                .reduce(partial(aggregate, dataset_chunk_slices), seed=0)
                .do_action(
                    partial(datasource_manager.copy, dataset_chunk, destination=datasource_manager.output_datasource,
                            slices=dataset_chunk_slices)
                )
            )
        )
        .reduce(lambda x, y: dataset_chunk, seed=dataset_chunk).map(lambda _: dataset_chunk)
    )

