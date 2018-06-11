import pytest
from rx import Observable

from chunkflow.block_processor import BlockProcessor
from chunkflow.models import Block


class TestBlockProcessor:

    def test_process_single_channel_2x2(self):
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

        block_processor = BlockProcessor()
        with pytest.raises(ValueError):
            block_processor.process(block, task_stream)

        assert block_processor.error is not None
