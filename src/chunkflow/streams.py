import multiprocessing
from collections import namedtuple
from functools import partial

from rx import Observable, config
from rx.concurrency import ThreadPoolScheduler
from rx.core.blockingobservable import BlockingObservable
from rx.internal import extensionmethod

optimal_thread_count = multiprocessing.cpu_count()
scheduler = ThreadPoolScheduler(optimal_thread_count)


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
    # Account for additional output dimensions
    slices = (slice(None),) * (len(datasource.shape) - len(slices)) + slices

    if hasattr(aggregate, '__getitem__'):
        aggregate[slices] += datasource[slices]
    else:
        # no slicing when seeding with 0
        aggregate += datasource[slices]
    return aggregate


def create_download_stream(executor, block, datasource_manager):
    return lambda chunk: Observable.just(chunk).do_action(datasource_manager.download_input)
    # return lambda chunk: Observable.just(chunk).flat_map(
    #     lambda s: executor.submit(datasource_manager.download_input, s))


def create_inference_stream(block, inference_operation, blend_operation, datasource_manager):
    return lambda chunk: (
        Observable.just(chunk)
        .map(inference_operation)
        .map(blend_operation)
        .do_action(datasource_manager.dump_chunk)
        .do_action(block.checkpoint)
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


def create_upload_stream(block, datasource_manager):
    return lambda chunk: (
        Observable.merge(
            # core slices can bypass to the final datasource
            Observable.just(chunk).map(block.core_slices).map(
                lambda slices: DumpArguments(datasource_manager.output_datasource_final, slices)
            ),
            Observable.just(chunk).flat_map(block.overlap_slices).map(
                lambda slices: DumpArguments(datasource_manager.output_datasource, slices)
            ),
        )
        .do_action(lambda dump_args: datasource_manager.dump_chunk(chunk, **dump_args._asdict()))
    )


def create_inference_and_blend_stream(executor, block, inference_operation, blend_operation, datasource_manager):
    return lambda chunk: (
        Observable.just(chunk)
        # .observe_on(scheduler)
        .flat_map(create_download_stream(executor, block, datasource_manager))
        .flat_map(create_inference_stream(block, inference_operation, blend_operation, datasource_manager))
        # check both the current chunk we just ran inference on as well as the neighboring chunks
        .flat_map(lambda chunk: Observable.from_(block.get_all_neighbors(chunk)).start_with(chunk))
        .filter(block.is_checkpointed)
        .filter(block.all_neighbors_checkpointed)
        .distinct()
        .flat_map(create_aggregate_stream(block, datasource_manager))
        .flat_map(create_upload_stream(block, datasource_manager))
        .map(lambda _: chunk)
    )


def create_blend_stream(block, datasource_manager):
    """
    Assume block is a dataset with chunks to represent each task!
    """
    return lambda chunk: (
        Observable.just(chunk)
        .flat_map(block.overlap_chunk_slices)
        .flat_map(
            lambda chunk_slices:
            (
                # create temp list of repositories values at time of iteration
                Observable.from_(list(datasource_manager.repository.overlap_datasources.values()))
                .reduce(partial(aggregate, chunk_slices), seed=0)
                .do_action(
                    partial(chunk.copy_data, destination=datasource_manager.output_datasource, slices=chunk_slices)
                )
            )
        )
        .map(lambda _: chunk)
    )
