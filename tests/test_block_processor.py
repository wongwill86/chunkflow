import pytest

import numpy as np

from chunkflow.block_processor import BlockProcessor
from chunkflow.chunk_operations.blend_operation import AverageBlend
from chunkflow.chunk_operations.chunk_operation import ChunkOperation
from chunkflow.cloudvolume_datasource import CloudVolumeCZYX
from chunkflow.datasource_manager import DatasourceManager
from chunkflow.datasource_manager import DatasourceRepository
from chunkflow.global_offset_array import GlobalOffsetArray
from chunkflow.models import Block


class NumpyDatasource(DatasourceRepository):
    def __init__(self, output_shape,
                 output_datasource_core=None, output_datasource_overlap=None, *args, **kwargs):
        self.output_shape = output_shape
        super().__init__(output_datasource_core=output_datasource_core,
                         output_datasource_overlap=output_datasource_overlap,
                         *args, **kwargs)

        if self.output_datasource_core is None:
            self.output_datasource_core = self.create(None)

        if self.output_datasource_overlap is None:
            self.output_datasource_overlap = self.create(None)

    def create(self, mod_index, *args, **kwargs):
        offset = self.input_datasource.global_offset
        shape = self.input_datasource.shape

        if len(self.output_shape) > len(self.input_datasource.shape):
            extra_dimensions = len(self.output_shape) - len(self.input_datasource.shape)

            shape = self.output_shape[0:extra_dimensions] + shape
            offset = (0,) * extra_dimensions + offset

        return GlobalOffsetArray(np.zeros(shape), global_offset=offset)


class IncrementInference(ChunkOperation):
    def __init__(self, step=1, *args, **kwargs):
        self.step = step
        super().__init__(*args, **kwargs)

    def _process(self, chunk):
        self.run_inference(chunk)

    def run_inference(self, chunk):
        chunk.data += self.step


class IncrementThreeChannelInference(ChunkOperation):
    def __init__(self, step=1, *args, **kwargs):
        self.step = step
        super().__init__(*args, **kwargs)

    def _process(self, chunk):
        self.run_inference(chunk)

    def run_inference(self, chunk):
        chunk.data += self.step
        global_offset = (0,) + chunk.data.global_offset
        one = chunk.data
        two = chunk.data * 10
        three = chunk.data * 100
        chunk.data = GlobalOffsetArray(np.stack((one, two, three)), global_offset=global_offset)


class TestBlockProcessor:

    def test_process_single_channel(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        output_shape = (3, 3)
        overlap = (1, 1)

        import numpy as np
        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=output_shape))
        processor = BlockProcessor(
            inference_operation=IncrementInference(step=1),
            blend_operation=AverageBlend(block),
            datasource_manager=datasource_manager
        )

        processor.process(block)

        assert datasource_manager.repository.output_datasource_core.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum() == \
            np.product(block.shape)

    def test_process_multi_channel(self):
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

        assert datasource_manager.repository.output_datasource_core.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum() == \
            np.product(block.shape) * 111

    # @pytest.mark.usefixtures('cloud_volume')
    # def test_process_cloudvolume(self, cloud_volume):
    #     # sestup data
    #     input_cloudvolume = CloudVolumeCZYX(INPUT_SINGLE_CHANNEL_PATH, cache=True)
    #     output_cloudvolume_core = CloudVolumeCZYX(INPUT_SINGLE_CHANNEL_PATH, cache=True)
    #     output_cloudvolume_overlap = CloudVolumeCZYX(INPUT_SINGLE_CHANNEL_PATH, cache=True)
    #     repository = CloudVolumeDatasourceRepository(input_cloudvolume, output_cloudvolume_core,
    #                                                  output_cloudvolume_overlap)

    #     assert repository.get_datasource((1, 2, 0)) == datasource
    #     # return
    #     bounds = (slice(0, 7), slice(0, 7))
    #     chunk_shape = (3, 3)
    #     overlap = (1, 1)

    #     import numpy as np
    #     block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

    #     fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
    #     datasource_manager = DatasourceManager(repository)

    #     processor = BlockProcessor(
    #         IncrementThreeChannelInference(step=1), AverageBlend(block), datasource_manager
    #     )

    #     processor.process(block)

    #     # assert
    #     #     np.product(block.shape) * 111, datasource_manager.repository.output_datasource_core.sum() +
    #     #     datasource_manager.repository.output_datasource_overlap.sum()
    #     # )
