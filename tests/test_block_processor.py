import unittest

from chunkflow.block_processor import BlockProcessor
from chunkflow.chunk_operations.blend_operation import AverageBlend
from chunkflow.chunk_operations.inference_operation import IncrementInference
from chunkflow.datasource_manager import DatasourceManager
from chunkflow.datasource_manager import NumpyDatasource
from chunkflow.models import Block


class BlockProcessorTest(unittest.TestCase):

    def test_process(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        import numpy as np
        block = Block(bounds, chunk_shape, overlap)

        fake_data = np.zeros(block.shape)
        datasource_manager = DatasourceManager(NumpyDatasource(fake_data))
        processor = BlockProcessor(
            IncrementInference(step=1), AverageBlend(block), datasource_manager
        )

        processor.process(block)

        self.assertEquals(
            np.product(block.shape), datasource_manager.repository.output_datasource_core.sum() +
            datasource_manager.repository.output_datasource_overlap.sum()
        )
        # print(datasource_manager.repository.output_datasource_core.sum())
        # print(datasource_manager.repository.output_datasource_overlap.sum())
        # print(datasource_manager.repository.output_datasource_core)
        # print(datasource_manager.repository.output_datasource_overlap)
