import traceback
from datetime import datetime
from functools import partial
from threading import current_thread

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
    # Account for additional output dimensions
    slices = (slice(None),) * (len(datasource.shape) - len(slices)) + slices

    if hasattr(aggregate, '__getitem__'):
        aggregate[slices] += datasource[slices]
    else:
        # no slicing when seeding with 0
        aggregate += datasource[slices]
    return aggregate


class BlockProcessor:
    def __init__(self, inference_operation, blend_operation, datasource_manager):
        self.inference_operation = inference_operation
        self.blend_operation = blend_operation
        self.datasource_manager = datasource_manager

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
            .do_action(self.datasource_manager.download_input)
            .map(self.inference_operation)
            .map(self.blend_operation)
            .do_action(self.datasource_manager.dump_chunk)
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
                    # create temp list of repositories values at time of iteration
                    Observable.from_(list(self.datasource_manager.repository.intermediate_datasources.values()))
                    .reduce(partial(aggregate, chunk.slices), seed=0)
                    .do_action(chunk.load_data)
                    .map(lambda _: chunk)
                )
            )
            .flat_map(lambda chunk:
                      Observable.merge(
                          Observable.just(chunk).flat_map(block.overlap_slices).do_action(
                              partial(self.datasource_manager.upload_output_overlap, chunk)),
                          Observable.just(chunk) \
                          .map(block.core_slices).do_action(
                              partial(self.datasource_manager.upload_output_core, chunk))
                      )
                      .map(lambda _: chunk)
                      )
            .subscribe(
                self.print_done,
                on_error=self.on_error
            )
        )

    def on_error(self, error):
        print('error error *&)*&*&)*\n\n')
        traceback.print_exception(None, error, error.__traceback__)
        raise error

    def print_done(self, chunk, data=None):
        print('****** %s--%s %s done ' % (datetime.now(), current_thread().name, chunk.unit_index,))
