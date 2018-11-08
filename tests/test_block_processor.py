import pytest
from chunkblocks.models import Block
from rx import Observable
from functools import reduce

from chunkflow.block_processor import BlockProcessor


def mock_inference_stream(block, completed):
    seed = set()
    return lambda chunk: (
        Observable.just(chunk)
        .do_action(lambda chunk:
                    completed.add(chunk.unit_index))
        .flat_map(lambda chunk: block.get_all_neighbors(chunk))
        .filter(lambda chunk: all([neighbor.unit_index in completed for neighbor in block.get_all_neighbors(chunk)]))
        .distinct_hash(lambda chunk: chunk.unit_index, seed=seed)
    )


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

