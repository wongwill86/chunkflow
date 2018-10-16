from collections import namedtuple
from enum import Enum
from functools import partial

import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from rx import Observable, config
from rx.core import AnonymousObservable
from rx.core.blockingobservable import BlockingObservable
from rx.internal import extensionmethod

from chunkflow.chunk_buffer import CacheMiss

MAX_RETRIES = 10

@extensionmethod(Observable)
def from_item_or_future(item_or_future, default=None):
    """
    Checks the item to see if it is a future-like. (could be asyncio future which is not part of the concurrent.futures
    package).
    """
    if hasattr(item_or_future, 'result'):  # should probably check if it is callable too
        return Observable.from_future(item_or_future)
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


def aggregate(slices, aggregate, datasource):
    # Account for additional output dimensions
    channel_dimensions = len(datasource.shape) - len(slices)
    channel_slices = (slice(None),) * (channel_dimensions) + slices

    try:
        data = datasource[channel_slices]
    except CacheMiss:
        data = datasource.get_item(channel_slices, fill_missing=True)

    # 0 from seed
    if aggregate is 0:
        slice_shape = tuple(s.stop - s.start for s in slices)
        offset = (0,) * channel_dimensions + tuple(s.start for s in slices)

        aggregate = GlobalOffsetArray(
            np.zeros(data.shape[0:channel_dimensions] + slice_shape, dtype=data.dtype),
            global_offset=offset
        )
        aggregate += data
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
        .map(inference_operation)
        .map(blend_operation)
        .do_action(lambda chunk: datasource_manager.dump_chunk(
            chunk, datasource=datasource_manager.overlap_repository.get_datasource(chunk.unit_index), use_executor=False
        ))
    )


def create_aggregate_stream(block, datasource_manager):
    return lambda chunk: (
        # sum the different datasources together
        Observable.just(chunk)
        .flat_map(
            lambda chunk:
            (
                # create temp list of repositories values at time of iteration
                Observable.from_(datasource_manager.overlap_repositories())
                .reduce(partial(aggregate, chunk.slices), seed=0)
                .do_action(chunk.load_data)
                .map(lambda _: chunk)
            )
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
    )


def create_checkpoint_observable(block, stage):
    return lambda chunk: (
        Observable.just(chunk)
        .do_action(lambda chunk: block.checkpoint(chunk, stage=stage.value))
        # check both the current chunk we just ran inference on as well as the neighboring chunks
        .flat_map(lambda chunk: Observable.from_(block.get_all_neighbors(chunk)).start_with(chunk))
        .filter(lambda chunk: block.is_checkpointed(chunk, stage=stage.value))
        .filter(lambda chunk: block.all_neighbors_checkpointed(chunk, stage=stage.value))
        .distinct_hash(key_selector=lambda c: c.unit_index, seed=stage.hashset)
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
            .filter(lambda datasource_chunk: block.all_checkpointed(
                block.slices_to_chunks(datasource_chunk.slices), stage=stage_to_check.value))
            .distinct_hash(key_selector=lambda c: c.unit_index, seed=stage_to_complete.hashset)
            .flat_map(
                lambda datasource_chunk:
                Observable.from_([datasource_manager.output_datasource, datasource_manager.output_datasource_final])
                .map(partial(datasource_manager.flush, datasource_chunk))
                .flat_map(Observable.from_item_or_future)
            )
            .reduce(lambda x, y: uploaded_chunk, seed=uploaded_chunk)  # reduce to wait for all to complete transferring
            .map(lambda _: uploaded_chunk)
            .retry(MAX_RETRIES)
        )
    else:
        return lambda uploaded_chunk: Observable.just(uploaded_chunk)


def create_inference_and_blend_stream(block, inference_operation, blend_operation, datasource_manager):
    class Stages(Enum):
        INFERENCE_DONE, UPLOAD_DONE, FLUSH_DONE = range(3)

        def __init__(self, *args, **kwargs):
            self.hashset = set()

    return lambda chunk: (
        Observable.just(chunk)
        .do_action(lambda chunk: print('Start ', chunk.unit_index))
        .flat_map(create_input_stream(datasource_manager))
        .do_action(lambda chunk: print('Finish Download ', chunk.unit_index))
        .flat_map(create_inference_stream(block, inference_operation, blend_operation, datasource_manager))
        .do_action(lambda chunk: print('Finish Inference ', chunk.unit_index))
        .flat_map(create_checkpoint_observable(block, Stages.INFERENCE_DONE))
        .flat_map(create_aggregate_stream(block, datasource_manager))
        .do_action(lambda chunk: print('Finish Aggregate ', chunk.unit_index))
        .flat_map(create_upload_stream(block, datasource_manager))
        .flat_map(create_checkpoint_observable(block, Stages.UPLOAD_DONE))
        .do_action(lambda chunk: datasource_manager.overlap_repository.clear(chunk.unit_index))
        .do_action(partial(datasource_manager.clear_buffer, datasource_manager.input_datasource))
        .do_action(lambda chunk: datasource_manager.clear_buffer(datasource_manager.input_datasource, chunk))

        .flat_map(create_flush_datasource_observable(datasource_manager, block, Stages.UPLOAD_DONE, Stages.FLUSH_DONE))
        .do_action(lambda chunk: print('Finish Flushing ', chunk.unit_index))
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
