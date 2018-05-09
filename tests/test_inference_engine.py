import multiprocessing
import unittest

from rx import Observable
from rx.concurrency import ThreadPoolScheduler

from chunkflow import iterators
# from rx.subjects import Subject


def run_inference(index):
    print('\trunning inference at index %s' % (index,))

def blend(index):
    print('---- blending at %s' % (index,))

def create_blend(index):
    return lambda x: blend(index)

def inference_done(index):
    return index in done

def check_blend(neighbors, index):
    neighbors = [neighbor for neighbor in neighbors if inside_bounds(neighbor)]
    # Observable.from_list(neighbors).all(inference_done).
    if any(neighbor not in done for neighbor in neighbors):
        print('\t\tNot running blend at %s neighbors: %s done: %s' % (index, neighbors, done))
        return Observable.from_list(neighbors)
    else:
        # blend(index)
        print('\t\t****** Running blend at %s neighbors: %s ' % (index, neighbors,))
        blended.append(index)
        return Observable.empty()

def check_inference(x):
    if x not in done:
        RuntimeError
        print('\trunning inference at index %s' % (x,))
        done.add(x)
    else:
        print('\tAlready ran inference %s' % (x,))
    return x

def recurse(index):
    # print('recursing at %s ' % (index, ))
    neighbor_stream = (
        Observable.just(index)
        .do_action(lambda x: print('recurse observing: %s' % (x,)))
        .map(iterators.get_all_neighbors)
        # .map(lambda x: [neighbor for neighbor in x if inside_bounds(neighbor)])
        .map(lambda x: check_blend(x, index))
        .flat_map(lambda x: x)
        # .filter(inside_bounds)
    )

    count[0] = count[0] + 1
    if count[0] > 63:
        print('BLAHALSDFJKLSD:FJLKSFDJKLSFKJLSFJKLJKLF')
        return Observable.empty()
    return neighbor_stream.flat_map(recurse)

optimal_thread_count = multiprocessing.cpu_count()
pool_scheduler = ThreadPoolScheduler(optimal_thread_count)
parents = set()

done = set()
blended = []
count = [0]

class InferenceEngineTest(unittest.TestCase):
    # def test_blah(self):
    #     parents.clear()
    #     start = (0, 0)
    #     parents.add(start)
    #     blah_stream = Observable.just(start).flat_map(recurse)#.publish()
    #     blah_stream.subscribe(lambda x: print('sub %s' % (x,)))
    #     # blah_stream.connect()
    #     print(done)
    #     print(len(done))
    #     print(set(done))
    #     print(len(set(done)))
    #     print(blended)
    #     print(len(blended))
    #     print(set(blended))
    #     print(len(set(blended)))
    #     import time
    #     # assert False

    # def test_blah2(self):
    #     start = (0, 0)
    #     src = Observable.just(start)

    #     def get_inference_stream():
    #         inference_stream = Subject()
    #         # inference_stream = (
    #         (
    #             inference_stream
    #             # Observable.just(index)
    #             # .filter(inside_bounds)
    #             .filter(lambda x: x not in done)
    #             .do_action(run_inference)
    #             .do_action(lambda x: print('adding to done %s' % (x,)) or done.add(x))
    #             .do_action(lambda x: blend_stream.on_next(x))
    #             .flat_map(get_all_neighbors)
    #             .map(lambda x: inference_stream.on_next(x))
    #             .subscribe(print)
    #         )
    #         # inference_stream.subscribe(print)
    #         return inference_stream

    #     blend_stream = Subject()
    #     (
    #         blend_stream
    #         .map(get_all_neighbors)
    #         .do_action(lambda x: print('here is neigh %s' % (x,)))
    #         .flat_map(lambda x:
    #                   Observable.if_then(lambda: not any(outside_bounds(neighbor) for neighbor in x),
    #                                      Observable.from_list(x),
    #                                      Observable.empty())
    #              )
    #         .subscribe(lambda x: get_inference_stream().on_next(x))
    #         # .flat_map(lambda x:
    #         #           Observable.if_then(lambda: any(outside_bounds(neighbor) for neighbor in x),
    #         #                              Observable.from_list(x),
    #         #                              Observable.empty)
    #         # )
    #         # .filter(inside_bounds)
    #         # .do_action(lambda x: Observable.just('sadfkjsadfkld;jsl;')
    #         # .subscribe(print)
    #         # .subscribe(lambda x: print('hi'))
    #     )


    #     # blend_stream.on_next((0,0))

    #     get_inference_stream().on_next(start)
    #     # src.subscribe(get_inference_stream)
    #     # assert False

    # def test_blah3(self):
    #     start = (0, 0)

    #     src = Observable.just(start)
    #     blend_stream = Subject()
    #     (
    #         blend_stream
    #         .do_action(lambda x: print('...blend_stream called with %s' % (x,)))
    #         .map(lambda x: ([neighbor for neighbor in get_all_neighbors(x) if inside_bounds(neighbor)], x))
    #         .do_action(lambda x: print('here is neigh %s' % (x,)))
    #         .flat_map(lambda x:
    #                   Observable.from_list(x[0]) if any(neighbor not in done for neighbor in x[0]) else
    #                   blend(x[1]) or Observable.empty()
    #                   )
    #         .filter(inside_bounds)
    #         .filter(lambda x: x not in done)
    #         .subscribe(lambda x: inference_stream.on_next(x))
    #     )
    #     inference_stream = Subject()
    #     (
    #         inference_stream
    #         .do_action(lambda x: print('inf got %s' % (x,)))
    #         .do_action(run_inference)
    #         .do_action(lambda x: done.add(x))
    #         .subscribe(lambda x: blend_stream.on_next(x))
    #     )

    #     print('runnin')
    #     done.clear()
    #     print(done)

    #     inference_stream.on_next(start)


    # def test_bfs(self):
    #     start = (0, 0)
    #     enqueue = Subject()
    #     dequeue = Subject()
    #     visited = set()
    #     done.clear()
    #     inference_stream = Subject()


    #     (
    #         enqueue
    #         .do_action(inference_stream)
    #         .do_action(lambda x: visited.add(x))
    #         .do_action(lambda x: print('got to enqueue %s visited: %s' % (x, visited)))
    #         .flat_map(get_all_neighbors)
    #         .filter(lambda x: x not in visited)
    #         .filter(inside_bounds)
    #         .subscribe(enqueue)
    #         # .subscribe(lambda x: enqueue.on_next(x))
    #         # .subscribe(inference_stream)
    #     )

    #     inference_stream.do_action(run_inference).map(
    #         lambda x: (
    #             x,
    #             get_all_neighbors(x)
    #             # Observable.from_list(get_all_neighbors(x)).filter(lambda x: x not in visited).filter(inside_bounds)
    #             )
    #     ).filter(
    #         # lambda x: x[1].all(lambda y: False and print('hello %s, visited is %s' % (x, visited)) and y in visited)
    #         lambda x: all([neighbor in visited for neighbor in x[1] if inside_bounds(neighbor)])
    #     ).subscribe(lambda x: blend(x[0]))

    #     # dequeue.subscribe(print)


    #     # src = Observable.just(start).subscribe(enqueue)
    #     enqueue.on_next(start)

    def test_with_iter(self):
        start = (0, 0)

        # bounds = (slice(0, 70), slice(0, 70))
        # block_size = (30, 30)
        # overlap = (10, 10)
        done = set()

        def inside_bounds(index):
            return not any(idx < 0 or idx > 3 for idx in index)

        (
            Observable.from_iterable(iterators.bfs_iterator(start, (4, 4)))
            .do_action(run_inference)
            .do_action(lambda x: done.add(x))
            .flat_map(iterators.get_all_neighbors)
            .filter(inside_bounds)
            .map(
                lambda x: (
                    x,
                    iterators.get_all_neighbors(x)
                    # Observable.from_list(get_all_neighbors(x)).filter(lambda x: x not in visited).filter(inside_bounds)
                )
            ).filter(
                # lambda x: x[1].all(lambda y: False and print('hello %s, visited is %s' % (x, visited)) and y in visited)
                lambda x: print("x %s and done %s, get %s" % (x, done, [
                    neighbor in done for neighbor in x[1] if inside_bounds(neighbor)])) or all([
                    neighbor in done for neighbor in x[1] if inside_bounds(neighbor)])
            ).subscribe(lambda x: blend(x[0]))
        )

        assert False
