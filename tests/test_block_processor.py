import numpy as np

from chunkflow.block_processor import BlockProcessor
from chunkflow.chunk_operations.blend_operation import AverageBlend
from chunkflow.chunk_operations.chunk_operation import ChunkOperation
from chunkflow.datasource_manager import DatasourceManager
from chunkflow.datasource_manager import DatasourceRepository
from chunkflow.cloudvolume_datasource import CloudVolumeDatasourceRepository
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
    def __init__(self, step=1, output_dtype=np.float64, *args, **kwargs):
        self.step = step
        self.output_dtype = output_dtype
        super().__init__(*args, **kwargs)

    def _process(self, chunk):
        self.run_inference(chunk)

    def run_inference(self, chunk):
        chunk.data = chunk.data.astype(self.output_dtype)
        chunk.data += self.step


class IncrementThreeChannelInference(ChunkOperation):
    def __init__(self, step=1, output_dtype=np.float64, *args, **kwargs):
        self.step = step
        self.output_dtype = output_dtype
        super().__init__(*args, **kwargs)

    def _process(self, chunk):
        self.run_inference(chunk)

    def run_inference(self, chunk):
        chunk.data = chunk.data.astype(self.output_dtype)
        chunk.data += self.step

        global_offset = chunk.data.global_offset

        one = chunk.data
        two = chunk.data * 10
        three = chunk.data * 100
        new_data = np.stack((one, two, three)).squeeze()

        if len(new_data.shape) > len(chunk.data.global_offset):
            global_offset = (0,) + chunk.data.global_offset
        chunk.data = GlobalOffsetArray(new_data, global_offset=global_offset)


class TestBlockProcessor:

    def test_process_single_channel(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        output_shape = (3, 3)
        overlap = (1, 1)

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

        assert np.product(block.shape) == \
            datasource_manager.repository.output_datasource_core.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum()

    def test_process_multi_channel(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=(3,) + fake_data.shape))

        processor = BlockProcessor(
            IncrementThreeChannelInference(step=1), AverageBlend(block), datasource_manager
        )

        processor.process(block)

        assert np.product(block.shape) * 111 == \
            datasource_manager.repository.output_datasource_core.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum()

    def test_process_single_channel_3d(self):
        bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=fake_data.shape))

        processor = BlockProcessor(
            IncrementInference(step=1), AverageBlend(block), datasource_manager
        )

        processor.process(block)

        assert np.product(block.shape) == \
            datasource_manager.repository.output_datasource_core.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum()

    def test_process_multi_channel_3d(self):
        bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=(3,) + fake_data.shape))

        processor = BlockProcessor(
            IncrementThreeChannelInference(step=1), AverageBlend(block), datasource_manager
        )

        processor.process(block)

        assert np.product(block.shape) * 111 == \
            datasource_manager.repository.output_datasource_core.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum()

    def test_process_cloudvolume(self, cloudvolume_factory):
        bounds = (slice(200, 207), slice(100, 107), slice(50, 57))
        chunk_shape = (3, 3, 3)
        overlap = (1, 1, 1)
        input_data_type = 'uint8'
        output_data_type = 'float32'

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)
        # Setup data
        input_cloudvolume = cloudvolume_factory.create('input', data_type=input_data_type, volume_size=block.shape,
                                                       chunk_size=chunk_shape)
        output_cloudvolume_core = cloudvolume_factory.create('output_core', data_type=output_data_type,
                                                             volume_size=block.shape, chunk_size=chunk_shape,
                                                             num_channels=3)
        output_cloudvolume_overlap = cloudvolume_factory.create('output_overlap', data_type=output_data_type,
                                                                volume_size=block.shape, chunk_size=chunk_shape,
                                                                num_channels=3)

        repository = CloudVolumeDatasourceRepository(input_cloudvolume, output_cloudvolume_core,
                                                     output_cloudvolume_overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape, dtype=np.dtype(input_data_type)),
                                      global_offset=(0,) * len(block.shape))
        input_cloudvolume[bounds] = fake_data

        datasource_manager = DatasourceManager(repository)

        processor = BlockProcessor(
            IncrementThreeChannelInference(step=1, output_dtype=np.dtype(output_data_type)),
            AverageBlend(block), datasource_manager
        )

        processor.process(block)

        print(output_cloudvolume_overlap[bounds])
        print(output_cloudvolume_core[bounds])
        assert False

        # assert
        #     np.product(block.shape) * 111, datasource_manager.repository.output_datasource_core.sum() +
        #     datasource_manager.repository.output_datasource_overlap.sum()
        # )
