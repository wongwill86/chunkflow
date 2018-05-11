import multiprocessing
import unittest

from rx import Observable
from rx.concurrency import ThreadPoolScheduler
from rx.core.blockingobservable import BlockingObservable
from rx import config
from rx.internal import extensionmethod

from chunkflow import iterators
from threading import current_thread

# from rx.subjects import Subject

inference = []
blend = []
edge_upload = []
inner_upload = []
clear = []

def run_inference(index):
    print("inference %s, %s" % (index, current_thread().name))
    inference.append(index)


def run_blend(index):
    print("-----blend %s, %s" % (index, current_thread().name))
    blend.append(index)


def run_edge_upload(index):
    print("edge upload %s, %s" % (index, current_thread().name))
    edge_upload.append(index)


def run_inner_upload(index):
    print("inner upload %s, %s" % (index, current_thread().name))
    inner_upload.append(index)


def run_clear(index):
    print("clear %s, %s" % (index, current_thread().name))
    clear.append(index)


class InferenceEngineTest(unittest.TestCase):

    @extensionmethod(BlockingObservable)
    def blocking_subscribe(source, on_next = None, on_error = None, on_completed = None):
        """
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

    def test_with_iter(self):
        optimal_thread_count = multiprocessing.cpu_count()
        scheduler = ThreadPoolScheduler(optimal_thread_count)
        start = (0, 0)

        # bounds = (slice(0, 70), slice(0, 70))
        # block_size = (30, 30)
        # overlap = (10, 10)
        done = set()

        def all_neighbors_done(index):
            return all(neighbor in done for neighbor in iterators.get_all_neighbors(index))

        def is_volume_edge(index):
            return any(idx == 0 or idx == 2 for idx in index)

        neighbor_stream = (
            Observable.from_(iterators.bfs_iterator(start, (3, 3)))
            .do_action(run_inference)
            .do_action(lambda x: done.add(x))
            .flat_map(iterators.get_all_neighbors)
            .flat_map(lambda x: Observable.just(x, scheduler=scheduler))
        )

        edge_stream, inner_stream = neighbor_stream.partition(is_volume_edge)

        edge_stream = (
            edge_stream.distinct().do_action(run_edge_upload)
        )

        inner_stream = (
            inner_stream
            .filter(all_neighbors_done)
            .distinct()
            .do_action(run_blend)
            .do_action(run_inner_upload)
        )

        Observable.merge(edge_stream, inner_stream).to_blocking().blocking_subscribe(run_clear)
        print('All done')

        print('inference: len: %s \t %s' % (len(inference), inference))
        print('blend: len: %s \t %s' % (len(blend), blend))
        print('edge_upload: len: %s \t %s' % (len(edge_upload), edge_upload))
        print('inner_upload: len: %s \t %s' % (len(inner_upload), inner_upload))
        print('clear: len: %s \t %s' % (len(clear), clear))
        assert False
