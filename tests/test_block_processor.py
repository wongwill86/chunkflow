import unittest

from chunkflow.block_processor import BlockProcessor
from chunkflow.chunk_operations.blend_operation import AverageBlend
from chunkflow.chunk_operations.inference_operation import IncrementInference
from chunkflow.chunk_operations.inference_operation import IncrementThreeChannelInference
from chunkflow.datasource_manager import DatasourceManager
from chunkflow.datasource_manager import NumpyDatasource
from chunkflow.global_offset_array import GlobalOffsetArray
from chunkflow.models import Block


class BlockProcessorTest(unittest.TestCase):

    def test_process(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        output_shape = (3, 3)
        overlap = (1, 1)

        import numpy as np
        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = DatasourceManager(NumpyDatasource(input_datasource=fake_data, output_shape=output_shape))
        processor = BlockProcessor(
            inference_operation=IncrementInference(step=1),
            blend_operation=AverageBlend(block),
            datasource_manager=datasource_manager
        )

        processor.process(block)

        print(datasource_manager.repository.output_datasource_core)
        print(datasource_manager.repository.output_datasource_overlap)
        self.assertEquals(
            np.product(block.shape), datasource_manager.repository.output_datasource_core.sum() +
            datasource_manager.repository.output_datasource_overlap.sum()
        )

    def test_process_multi_channel(self):
        # return
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        import numpy as np
        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=(3,) + fake_data.shape))

        processor = BlockProcessor(
            IncrementThreeChannelInference(step=1), AverageBlend(block), datasource_manager
        )

        processor.process(block)

        print(datasource_manager.repository.output_datasource_core)
        print(datasource_manager.repository.output_datasource_overlap)
        self.assertEquals(
            np.product(block.shape) * 111, datasource_manager.repository.output_datasource_core.sum() +
            datasource_manager.repository.output_datasource_overlap.sum()
        )
