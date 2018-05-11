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



class InferenceEngineTest(unittest.TestCase):


    @extensionmethod(BlockingObservable)
    def blocking_subscribe(source, on_next = None, on_error = None, on_completed = None):
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
            .observe_on(pool_scheduler)
            .flat_map(iterators.get_all_neighbors)
        )

        edge_stream, inner_stream = neighbor_stream.partition(is_volume_edge)

        blah = (
            inner_stream
            .filter(all_neighbors_done)
            .distinct()
            .do_action(blend)
            .do_action(upload)
        )

        boh = (
            edge_stream.distinct().do_action(upload)
        )

        Observable.merge(blah, boh).subscribe(clear)
        print('waiting to finish')
        # time.sleep(2)
        pool_scheduler.executor.shutdown()

        # assert False

    def test_ugh(self):
        optimal_thread_count = multiprocessing.cpu_count()
        scheduler = ThreadPoolScheduler(optimal_thread_count)
        print('optimal thread count %s' % optimal_thread_count)

        from rx import config
        self.latch = config['concurrency'].Event()
        # def on_completed():
        #     print('Completed...')
        #     self.latch.set()
        # Observable.from_(range(5)) \
        #     .zip(Observable.interval(1000), lambda x,y: x) \
        #     .subscribe_on(scheduler) \
        #     .blocking_subscribe(print, print, on_completed)
        # self.latch.wait()
        # Observable.range(1, 20) \
        # Observable.range(0, 100)\
        #     .select_many(lambda x: Observable.start(lambda: x, scheduler=scheduler)) \
        #     .map(lambda i: i * 100) \
        #     .observe_on(scheduler) \
        #     .map(intense_calculation) \
        #     .subscribe(on_next=lambda i: print("PROCESS 3: {0} {1}".format(current_thread().name, i)),
        #                on_error=lambda e: print(e))
        (
            Observable.from_(range(1, 10, 5))
            .do_action(intense_calculation2)
            # .select_many(lambda i: Observable.start(lambda: i, scheduler=scheduler))
            .observe_on(scheduler)
            .flat_map(lambda x: Observable.from_(range(x, x + 5), scheduler=scheduler))
            # .observe_on(scheduler)
            .do_action(intense_calculation)
            .do_action(intense_calculation3)
            .to_blocking()
            .blocking_subscribe(
                on_next=lambda x: printthread("on_next: {}".format(x)),
                on_completed=lambda: printthread("on_completed"),
                on_error=lambda err: printthread("on_error: {}".format(err)))
        )
        #     .map(lambda s: intense_calculation(s)) \
        #     .subscribe_on(pool_scheduler) \
        #     .subscribe(on_next=lambda i: print("PROCESS 2: {0} {1}".format(current_thread().name, i)),
        #             on_error=lambda e: print(e), on_completed=lambda: print("PROCESS 2 done!"))

        # Create Process 3, which is infinite
        # Observable.range(1, 100) \
        #     .map(lambda i: i * 100) \
        #     .observe_on(pool_scheduler) \
        #     .map(lambda s: intense_calculation(s)) \
        #     .subscribe(on_next=lambda i: print("PROCESS 3: {0} {1}".format(current_thread().name, i)),
        #             on_error=lambda e: print(e))

        printthread("\nAll done")
        # scheduler.executor.shutdown(wait=True)
        # time.sleep(4)
        # pool_scheduler.executor.shutdown()
        assert False

import random
import time
from rx.core import Scheduler

def printthread(val):
    print("{}, \tthread: {}".format(val, current_thread().name))

def intense_calculation2(value):
    printthread("pre caclc {}".format(value))
    # time.sleep(random.random() * 2)
    return value

def intense_calculation(value):
    printthread("\t\tcalc {}".format(value))
    time.sleep(random.random() * 1)
    return value

def intense_calculation3(value):
    printthread("\t\t\tpost calce {}".format(value))
    time.sleep(random.random() * 1)
