import unittest

from chunkflow.blend_engine import IdentityBlend
from chunkflow.block_processor import BlockProcessor
from chunkflow.datasource_manager import NumpyDatasource
from chunkflow.inference_engine import IdentityInference
from chunkflow.models import Block


class BlockProcessorTest(unittest.TestCase):

    def test_process(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_size = (30, 30)
        overlap = (10, 10)

        import numpy as np
        block = Block(bounds, chunk_size, overlap)

        processor = BlockProcessor(
            IdentityInference(factor=1), IdentityBlend(factor=1), NumpyDatasource(np.ones((100, 100)))
        )

        processor.process(block)
        # assert False
    # def test_with_iter(self):
    #     optimal_thread_count = multiprocessing.cpu_count()
    #     scheduler = ThreadPoolScheduler(optimal_thread_count)
    #     start = (0, 0)

    #     # bounds = (slice(0, 70), slice(0, 70))
    #     # chunk_size = (30, 30)
    #     # overlap = (10, 10)
    #     done = set()
    #     iterator = iterators.UnitBFSIterator()

    #     def all_neighbors_done(index):
    #         return all(neighbor in done for neighbor in iterator.get_all_neighbors(index))

    #     def is_volume_edge(index):
    #         return any(idx == 0 or idx == 2 for idx in index)

    #     inf = []
    #     blend = []
    #     neighbor_stream = (
    #         Observable.from_(iterator.iterator(start, (10, 10)))
    #         .do_action(lambda x: inf.append(x) or print('---------running inf %s' % (x,)))
    #         .do_action(lambda x: done.add(x))
    #         .flat_map(iterator.get_all_neighbors)
    #         .flat_map(lambda x: Observable.just(x, scheduler=scheduler))
    #     )

    #     edge_stream, inner_stream = neighbor_stream.partition(is_volume_edge)

    #     edge_stream = (
    #         edge_stream.distinct().do_action(lambda x: print('do edge %s' % (x,)))
    #     )

    #     inner_stream = (
    #         inner_stream
    #         .filter(all_neighbors_done)
    #         .distinct()
    #         .do_action(lambda x: blend.append(x) or print('***************blend %s' % (x,)))
    #         .do_action(lambda x: print('inner upoload %s' % (x,)))
    #     )

    #     # inner_stream.subscribe(print)
    #     # print('----finished egde')
    #     # edge_stream.subscribe(print)
    #     Observable.merge(edge_stream, inner_stream).subscribe(lambda x: print('clearing %s' % (x,)))
    #     print('All done')

    #     print('inference: len: %s \t %s' % (len(inf), inf))
    #     print('blend: len: %s \t %s' % (len(blend), blend))
    #     # print('edge_upload: len: %s \t %s' % (len(edge_upload), edge_upload))
    #     # print('inner_upload: len: %s \t %s' % (len(inner_upload), inner_upload))
    #     # print('clear: len: %s \t %s' % (len(clear), clear))
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
    #     chunk_size = (30, 30)
    #     bfs = deque(iterators.blocked_iterator(bounds, chunk_size, iterators.bfs_iterator))
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
    #     chunk_size = (30, 30)
    #     overlap = (10, 10)
    #     bfs = deque(iterators.blocked_iterator(bounds, chunk_size, iterators.bfs_iterator, overlap=overlap))
    #     for expected_bfs in expected_bfs_steps:
    #         while len(expected_bfs) and len(bfs):
    #             step = bfs.popleft()
    #             expected_bfs.remove(str(step))
    #         self.assertEquals(0, len(expected_bfs))

    #     self.assertEquals(0, len(bfs))
