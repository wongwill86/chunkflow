from collections import namedtuple
from enum import Enum
from functools import partial

import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from rx import Observable, config
from rx.core import AnonymousObservable
from rx.core.blockingobservable import BlockingObservable
from rx.internal import extensionmethod


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

    # 0 from seed
    if aggregate is 0:
        data = datasource[channel_slices]
        slice_shape = tuple(s.stop - s.start for s in slices)
        offset = (0,) * channel_dimensions + tuple(s.start for s in slices)

        aggregate = GlobalOffsetArray(
            np.zeros(data.shape[0:channel_dimensions] + slice_shape, dtype=data.dtype),
            global_offset=offset
        )
        aggregate += data
    else:
        aggregate[channel_slices] += datasource[channel_slices]

    return aggregate


def create_download_stream(block, datasource_manager, executor=None):
    if executor is None:
        return lambda chunk: Observable.just(chunk).do_action(datasource_manager.download_input)
    else:
        return lambda chunk: Observable.just(chunk).flat_map(
            lambda chunk: executor.submit(chunk.load_data, datasource_manager.input_datasource)
        )


def create_inference_stream(block, inference_operation, blend_operation, datasource_manager):
    return lambda chunk: (
        Observable.just(chunk)
        .map(inference_operation)
        .map(blend_operation)
        .do_action(datasource_manager.dump_chunk)
    )


def create_aggregate_stream(block, datasource_manager):
    return lambda chunk: (
        # sum the different datasources together
        Observable.just(chunk)
        .flat_map(
            lambda chunk:
            (
                # create temp list of repositories values at time of iteration
                Observable.from_(list(datasource_manager.repository.overlap_datasources.values()))
                .reduce(partial(aggregate, chunk.slices), seed=0)
                .do_action(chunk.load_data)
                .map(lambda _: chunk)
            )
        )
    )


DumpArguments = namedtuple('DumpArguments', 'datasource slices')


def create_upload_stream(block, datasource_manager, executor=None):
    def append_dump(observable, chunk):
        if executor is None:
            return observable.do_action(
                lambda dump_args: datasource_manager.dump_chunk(chunk, **dump_args._asdict())
            )
        else:
            return observable.flat_map(
                lambda dump_args: executor.submit(chunk.dump_args, **dump_args._asdict())
            )

    return lambda chunk: append_dump(
        Observable.merge(
            # core slices can bypass to the final datasource
            Observable.just(chunk).map(block.core_slices).map(
                lambda slices: DumpArguments(datasource_manager.output_datasource_final, slices)
            ),
            Observable.just(chunk).flat_map(block.overlap_slices).map(
                lambda slices: DumpArguments(datasource_manager.output_datasource, slices)
            )
        ),
        chunk
    ).reduce(lambda x, y: chunk, seed=chunk).map(lambda _: chunk)  # reduce to wait until all have completed transferring


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
    output_needs_flush = hasattr(datasource_manager.output_datasource, 'block')
    output_final_needs_flush = hasattr(datasource_manager.output_datasource_final, 'block')

    reference_datasource = (
        datasource_manager.output_datasource if output_needs_flush else
        datasource_manager.output_datasource_final if output_final_needs_flush else None
    )

    if reference_datasource is not None:
        return lambda uploaded_chunk: (
            Observable.from_(reference_datasource.block.slices_to_chunks(uploaded_chunk.slices))
            .filter(lambda datasource_chunk: block.all_checkpointed(
                block.slices_to_chunks(datasource_chunk.slices), stage=stage_to_check.value))
            .distinct_hash(key_selector=lambda c: c.unit_index, seed=stage_to_complete.hashset)
            .flat_map(lambda datasource_chunk:
                      Observable.merge(
                          Observable.empty() if not output_needs_flush else Observable.just(
                          datasource_manager.output_datasource),
                          Observable.empty() if not output_final_needs_flush else Observable.just(
                          datasource_manager.output_datasource_final)
                      ).do_action(lambda datasource: datasource.flush(unit_index=datasource_chunk.unit_index))
            )
            .reduce(lambda x, y: uploaded_chunk, seed=uploaded_chunk) # reduce to wait to all have finished transferring
            .map(lambda _: uploaded_chunk)
        )
    else:
        return lambda uploaded_chunk: Observable.just(uploaded_chunk)


def create_inference_and_blend_stream(block, inference_operation, blend_operation, datasource_manager, scheduler=None,
                                      io_executor=None):
    class Stages(Enum):
        INFERENCE_DONE, UPLOAD_DONE, FLUSH_DONE = range(3)

        def __init__(self, *args, **kwargs):
            self.hashset = set()

    return lambda chunk: (
        (Observable.just(chunk) if scheduler is None else Observable.just(chunk).observe_on(scheduler))
        .do_action(lambda chunk: print('Start ', chunk.unit_index))
        .flat_map(create_download_stream(block, datasource_manager, io_executor))
        .do_action(lambda chunk: print('Finish Download ', chunk.unit_index))
        .flat_map(create_inference_stream(block, inference_operation, blend_operation, datasource_manager))
        .do_action(lambda chunk: print('Finish Inference ', chunk.unit_index))
        .flat_map(create_checkpoint_observable(block, Stages.INFERENCE_DONE))

        .flat_map(create_aggregate_stream(block, datasource_manager))
        .do_action(lambda chunk: print('Finish Aggregate ', chunk.unit_index))
        .flat_map(create_upload_stream(block, datasource_manager, io_executor))
        .do_action(lambda chunk: print('Finish Upload ', chunk.unit_index))

        .flat_map(create_checkpoint_observable(block, Stages.UPLOAD_DONE))
        .do_action(datasource_manager.clear)

        .flat_map(create_flush_datasource_observable(datasource_manager, block, Stages.UPLOAD_DONE, Stages.FLUSH_DONE))
        .map(lambda _: chunk)
    )


def create_blend_stream(block, datasource_manager, scheduler=None):
    """
    Assume block is a dataset with chunks to represent each task!
    """
    return lambda chunk: (
        (Observable.just(chunk) if scheduler is None else Observable.just(chunk).observe_on(scheduler))
        .flat_map(block.overlap_chunk_slices)
        .flat_map(
            lambda chunk_slices:
            (
                # create temp list of repositories values at time of iteration
                Observable.from_(list(datasource_manager.repository.overlap_datasources.values()))
                .reduce(partial(aggregate, chunk_slices))
                .do_action(
                    partial(chunk.copy_data, destination=datasource_manager.output_datasource, slices=chunk_slices)
                )
            )
        )
        .map(lambda _: chunk)
    )
