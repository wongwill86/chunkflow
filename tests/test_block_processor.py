import pytest
from chunkblocks.models import Block
from rx import Observable
from functools import reduce

from chunkflow.block_processor import BlockProcessor, ReadyNeighborIterator


class TestBlockProcessor:

    def test_error_throw(self):
        return
        bounds = (slice(0, 5), slice(0, 5))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        def throws(val):
            if True:
                raise ValueError('test')
            else:
                return 0

        def task_stream(chunk):
            return Observable.just(chunk).map(throws)

        block_processor = BlockProcessor(block)
        with pytest.raises(ValueError):
            block_processor.process(task_stream)

        assert block_processor.error is not None

    def test_ready_iterator_integration(self):
        return
        bounds = (slice(0, 9), slice(0, 9))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        block_processor = BlockProcessor(block, on_next=lambda chunk: accumulate_next(chunk))
        block_processor.process(lambda chunk: Observable.just(chunk))

        assert reduce(lambda x, y: x * y, block.num_chunks) == len(completed)

    def test_ready_iterator_integration(self):
        return
        bounds = (slice(0, 9), slice(0, 9))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        done_iterator = ReadyNeighborIterator(block).get()
        done_iterator.send(None)
        completed = set()
        def accumulate_next(chunk):
            completed.add(chunk)

        block_processor = BlockProcessor(block,
                                         on_next=lambda chunk: accumulate_next(chunk) or done_iterator.send(chunk))
        block_processor.process(lambda chunk: Observable.just(chunk))

        assert reduce(lambda x, y: x * y, block.num_chunks) == len(completed)


class TestReadyNeighborIterator:
    def test_done_iterator(self):
        # bounds = (slice(0, 9), slice(0, 9))
        chunk_shape = (3, 3)
        overlap = (1, 1)
        offset = (0, 0)
        num_chunks = (10, 10)

        block = Block(offset=offset, num_chunks=num_chunks, chunk_shape=chunk_shape, overlap=overlap)

        done_iterator = ReadyNeighborIterator(block)
        it = done_iterator.get()
        processed = 0
        step_1 = set()
        step_2 = set()

        for chunk in it:
            step_1.add(chunk.unit_index)
            print('finished', chunk.unit_index, 'i believe are completed', step_1)

            for neighbor in done_iterator.get_all_neighbors(chunk.unit_index, block.num_chunks):
                print('checking ', neighbor, 'for ', list(neighbor_neighbor for neighbor_neighbor in done_iterator.get_all_neighbors(
                    neighbor, block.num_chunks)))
                if all(neighbor_neighbor in step_1 for neighbor_neighbor in done_iterator.get_all_neighbors(
                    neighbor, block.num_chunks)) and neighbor not in step_2:
                    step_2.add(neighbor)
                    try:
                        print('the start')
                        it.send(block.unit_index_to_chunk(neighbor))
                    except StopIteration:
                        pass
            processed += 1

        assert reduce(lambda x, y: x * y, block.num_chunks) == processed
