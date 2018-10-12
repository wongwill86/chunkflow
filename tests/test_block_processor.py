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
        bounds = (slice(0, 9), slice(0, 9))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        block_processor = BlockProcessor(block, on_next=lambda chunk: accumulate_next(chunk))
        block_processor.process(lambda chunk: Observable.just(chunk))

        assert reduce(lambda x, y: x * y, block.num_chunks) == len(completed)

    def test_ready_iterator_integration(self):
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
        bounds = (slice(0, 9), slice(0, 9))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        done_iterator = ReadyNeighborIterator(block)
        it = done_iterator.get()
        processed = 0
        completed = set()
        for chunk in it:
            completed.add(chunk.unit_index)
            if all(
                neighbor in completed for neighbor in done_iterator.get_all_neighbors(
                    chunk.unit_index, block.num_chunks)) and chunk.unit_index not in completed:
                it.send(chunk)
            processed += 1

        assert reduce(lambda x, y: x * y, block.num_chunks) == processed
