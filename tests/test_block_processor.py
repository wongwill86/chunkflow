import multiprocessing
import unittest
from threading import current_thread

from rx import Observable
from rx import config
from rx.concurrency import ThreadPoolScheduler
from rx.core.blockingobservable import BlockingObservable
from rx.internal import extensionmethod

from chunkflow import iterators
from chunkflow.blend_engine import IdentityBlend
from chunkflow.block_processor import BlockProcessor
from chunkflow.inference_engine import IdentityInference


class BlockProcessorTest(unittest.TestCase):

    def test_init_wrong_size_no_overlap(self):
        bounds = (slice(0, 70), slice(0, 70))
        block_size = (30, 30)

        processor = BlockProcessor(None, None, block_size)
        with self.assertRaises(ValueError):
            processor.process(bounds)

    def test_init_wrong_size_overlap(self):
        bounds = (slice(0, 70), slice(0, 70))
        block_size = (30, 30)
        overlap = (11, 11)

        processor = BlockProcessor(None, None, block_size, overlap)
        with self.assertRaises(ValueError):
            processor.process(bounds)

    def test_index_slices(self):
        bounds = (slice(0, 70), slice(0, 70))
        block_size = (30, 30)
        overlap = (10, 10)

        processor = BlockProcessor(None, None, block_size, overlap)

        self.assertEquals((slice(0, 30), slice(0, 30)), processor._unit_index_to_slices(bounds, (0, 0)))
        self.assertEquals((slice(0, 30), slice(20, 50)), processor._unit_index_to_slices(bounds, (0, 1)))
        self.assertEquals((slice(20, 50), slice(0, 30)), processor._unit_index_to_slices(bounds, (1, 0)))
        self.assertEquals((slice(20, 50), slice(20, 50)),processor._unit_index_to_slices(bounds, (1, 1)))

    def test_process(self):
        bounds = (slice(0, 70), slice(0, 70))
        block_size = (30, 30)
        overlap = (10, 10)

        processor = BlockProcessor(IdentityInference('test', 'blah'), IdentityBlend('test', 'blah'), block_size, overlap)
        processor.process(bounds)

        assert False

    # def test_with_iter(self):
    #     optimal_thread_count = multiprocessing.cpu_count()
    #     scheduler = ThreadPoolScheduler(optimal_thread_count)
    #     start = (0, 0)

    #     # bounds = (slice(0, 70), slice(0, 70))
    #     # block_size = (30, 30)
    #     # overlap = (10, 10)
    #     done = set()

    #     def all_neighbors_done(index):
    #         return all(neighbor in done for neighbor in iterators.get_all_neighbors(index))

    #     def is_volume_edge(index):
    #         return any(idx == 0 or idx == 2 for idx in index)

    #     neighbor_stream = (
    #         Observable.from_(iterators.bfs_iterator(start, (3, 3)))
    #         .do_action(run_inference)
    #         .do_action(lambda x: done.add(x))
    #         .flat_map(iterators.get_all_neighbors)
    #         .flat_map(lambda x: Observable.just(x, scheduler=scheduler))
    #     )

    #     edge_stream, inner_stream = neighbor_stream.partition(is_volume_edge)

    #     edge_stream = (
    #         edge_stream.distinct().do_action(run_edge_upload)
    #     )

    #     inner_stream = (
    #         inner_stream
    #         .filter(all_neighbors_done)
    #         .distinct()
    #         .do_action(run_blend)
    #         .do_action(run_inner_upload)
    #     )

    #     Observable.merge(edge_stream, inner_stream).to_blocking().blocking_subscribe(run_clear)
    #     print('All done')

    #     print('inference: len: %s \t %s' % (len(inference), inference))
    #     print('blend: len: %s \t %s' % (len(blend), blend))
    #     print('edge_upload: len: %s \t %s' % (len(edge_upload), edge_upload))
    #     print('inner_upload: len: %s \t %s' % (len(inner_upload), inner_upload))
    #     print('clear: len: %s \t %s' % (len(clear), clear))
    #     assert False

    # def test_blocked_iterator_no_overlap(self):
    #     expected_bfs_steps = [
    #         set([str((slice(0, 30), slice(0, 30)))]),
    #         set([
    #             str((slice(0, 30), slice(30, 60))),
    #             str((slice(30, 60), slice(0, 30))),
    #             str((slice(30, 60), slice(30, 60))),
    #         ]),
    #         set([
    #             str((slice(0, 30), slice(60, 90))),
    #             str((slice(30, 60), slice(60, 90))),
    #             str((slice(60, 90), slice(60, 90))),
    #             str((slice(60, 90), slice(30, 60))),
    #             str((slice(60, 90), slice(0, 30))),
    #         ])
    #     ]
    #     print(expected_bfs_steps)

    #     bounds = (slice(0, 90), slice(0, 90))
    #     block_size = (30, 30)
    #     bfs = deque(iterators.blocked_iterator(bounds, block_size, iterators.bfs_iterator))
    #     for expected_bfs in expected_bfs_steps:
    #         while len(expected_bfs) and len(bfs):
    #             step = bfs.popleft()
    #             expected_bfs.remove(str(step))
    #         self.assertEquals(0, len(expected_bfs))

    #     self.assertEquals(0, len(bfs))

    # def test_blocked_iterator_overlap(self):
    #     expected_bfs_steps = [
    #         set([str((slice(0, 30), slice(0, 30)))]),
    #         set([
    #             str((slice(0, 30), slice(20, 50))),
    #             str((slice(20, 50), slice(0, 30))),
    #             str((slice(20, 50), slice(20, 50))),
    #         ]),
    #         set([
    #             str((slice(0, 30), slice(40, 70))),
    #             str((slice(20, 50), slice(40, 70))),
    #             str((slice(40, 70), slice(40, 70))),
    #             str((slice(40, 70), slice(20, 50))),
    #             str((slice(40, 70), slice(0, 30))),
    #         ])
    #     ]
    #     print(expected_bfs_steps)

    #     bounds = (slice(0, 70), slice(0, 70))
    #     block_size = (30, 30)
    #     overlap = (10, 10)
    #     bfs = deque(iterators.blocked_iterator(bounds, block_size, iterators.bfs_iterator, overlap=overlap))
    #     for expected_bfs in expected_bfs_steps:
    #         while len(expected_bfs) and len(bfs):
    #             step = bfs.popleft()
    #             expected_bfs.remove(str(step))
    #         self.assertEquals(0, len(expected_bfs))

    #     self.assertEquals(0, len(bfs))
