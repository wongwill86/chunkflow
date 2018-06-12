import numpy as np
from rx import Observable

from chunkflow.chunk_operations.blend_operation import AverageBlend
from chunkflow.chunk_operations.chunk_operation import ChunkOperation
from chunkflow.cloudvolume_datasource import CloudVolumeDatasourceRepository
from chunkflow.datasource_manager import DatasourceManager
from chunkflow.datasource_manager import DatasourceRepository
from chunkflow.global_offset_array import GlobalOffsetArray
from chunkflow.iterators import UnitIterator
from chunkflow.models import Block
from chunkflow.streams import create_blend_stream
from chunkflow.streams import create_inference_and_blend_stream


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


class TestInferenceStream:

    def test_process_single_channel_2x2(self):
        bounds = (slice(0, 5), slice(0, 5))
        chunk_shape = (3, 3)
        output_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=output_shape))

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementInference(step=1),
            blend_operation=AverageBlend(block),
            datasource_manager=datasource_manager
        )

        test_chunk_0_0 = block.unit_index_to_chunk((0, 0))
        assert not block.is_checkpointed(test_chunk_0_0)
        Observable.just(test_chunk_0_0).flat_map(task_stream).subscribe(print)
        assert block.is_checkpointed(test_chunk_0_0)

        test_chunk_0_1 = block.unit_index_to_chunk((0, 1))
        assert not block.is_checkpointed(test_chunk_0_1)
        Observable.just(test_chunk_0_1).flat_map(task_stream).subscribe(print)
        assert block.is_checkpointed(test_chunk_0_1)

        test_chunk_1_0 = block.unit_index_to_chunk((1, 0))
        assert not block.is_checkpointed(test_chunk_1_0)
        Observable.just(test_chunk_1_0).flat_map(task_stream).subscribe(print)
        assert block.is_checkpointed(test_chunk_1_0)

        assert 0 == \
            datasource_manager.repository.output_datasource_core.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum()

        test_chunk_1_1 = block.unit_index_to_chunk((1, 1))
        assert not block.is_checkpointed(test_chunk_1_1)
        Observable.just(test_chunk_1_1).flat_map(task_stream).subscribe(print)
        assert block.is_checkpointed(test_chunk_1_1)

        assert np.product(block.shape) == \
            datasource_manager.repository.output_datasource_core.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum()

    def test_process_single_channel_3x3(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        output_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=output_shape))

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementInference(step=1),
            blend_operation=AverageBlend(block),
            datasource_manager=datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

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

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementThreeChannelInference(step=1),
            blend_operation=AverageBlend(block),
            datasource_manager=datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

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

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementInference(step=1),
            blend_operation=AverageBlend(block),
            datasource_manager=datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

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

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementThreeChannelInference(step=1),
            blend_operation=AverageBlend(block),
            datasource_manager=datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

        assert np.product(block.shape) * 111 == \
            datasource_manager.repository.output_datasource_core.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum()

    def test_process_cloudvolume(self, cloudvolume_factory):
        return
        bounds = (slice(200, 205), slice(100, 105), slice(50, 55))
        voxel_offset = (200, 100, 50)
        chunk_shape = (3, 3, 3)
        cloud_volume_chunk_size = (2, 2, 2)
        overlap = (1, 1, 1)
        input_data_type = 'uint8'
        output_data_type = 'float32'

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)
        # Setup data
        input_cloudvolume = cloudvolume_factory.create(
            'input', data_type=input_data_type, volume_size=block.shape, chunk_size=cloud_volume_chunk_size,
            voxel_offset=voxel_offset)
        output_cloudvolume_core = cloudvolume_factory.create(
            'output_core', data_type=output_data_type, volume_size=block.shape, chunk_size=cloud_volume_chunk_size,
            num_channels=3, voxel_offset=voxel_offset)
        output_cloudvolume_overlap = cloudvolume_factory.create(
            'output_overlap', data_type=output_data_type, volume_size=block.shape,
            chunk_size=cloud_volume_chunk_size, num_channels=3)

        repository = CloudVolumeDatasourceRepository(input_cloudvolume, output_cloudvolume_core,
                                                     output_cloudvolume_overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape, dtype=np.dtype(input_data_type)),
                                      global_offset=(0,) * len(block.shape))
        input_cloudvolume[(slice(None),) + bounds] = fake_data

        datasource_manager = DatasourceManager(repository)

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementThreeChannelInference(step=1, output_dtype=np.dtype(output_data_type)),
            blend_operation=AverageBlend(block),
            datasource_manager=datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

        assert np.product(block.shape) * 111 == \
            datasource_manager.repository.output_datasource_core[bounds].sum() + \
            datasource_manager.repository.output_datasource_overlap[bounds].sum()


class TestBlendStream:

    def test_blend(self):
        dataset_bounds = (slice(0, 7), slice(0, 7))
        task_size = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=dataset_bounds, chunk_shape=task_size, overlap=overlap)
        assert block.num_chunks == (3, 3)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))

        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=fake_data.shape))

        chunk_index = (1, 1)

        chunk = block.unit_index_to_chunk(chunk_index)

        # set up test data
        datasource_manager.repository.get_datasource(chunk.unit_index)[chunk.slices] = 1
        for neighbor in UnitIterator().get_all_neighbors(chunk_index):
            datasource_manager.repository.get_datasource(neighbor)[chunk.slices] = 1

        blend_stream = create_blend_stream(block, datasource_manager)

        Observable.just(chunk).flat_map(blend_stream).subscribe(print)

        assert 3 ** len(chunk_index) * 3 == \
            datasource_manager.output_datasource_core.sum()

    def test_blend_3d(self):
        dataset_bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        task_size = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=dataset_bounds, chunk_shape=task_size, overlap=overlap)
        assert block.num_chunks == (3, 3, 3)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))

        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=fake_data.shape))

        chunk_index = (1, 1, 1)

        chunk = block.unit_index_to_chunk(chunk_index)

        # set up test data
        datasource_manager.repository.get_datasource(chunk.unit_index)[chunk.slices] = 1
        for neighbor in UnitIterator().get_all_neighbors(chunk_index):
            datasource_manager.repository.get_datasource(neighbor)[chunk.slices] = 1

        blend_stream = create_blend_stream(block, datasource_manager)

        Observable.just(chunk).flat_map(blend_stream).subscribe(print)

        assert 3 ** len(chunk_index) * 7 == \
            datasource_manager.output_datasource_core.sum()

    def test_blend_multichannel_3d(self):
        dataset_bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        task_size = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=dataset_bounds, chunk_shape=task_size, overlap=overlap)
        assert block.num_chunks == (3, 3, 3)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        output_shape = (3,) + fake_data.shape

        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=output_shape))

        chunk_index = (1, 1, 1)

        chunk = block.unit_index_to_chunk(chunk_index)

        # set up test data
        datasource_manager.repository.get_datasource(chunk.unit_index)[(slice(None),) + chunk.slices] = 1
        for neighbor in UnitIterator().get_all_neighbors(chunk_index):
            datasource_manager.repository.get_datasource(neighbor)[(slice(None),) + chunk.slices] = 1

        blend_stream = create_blend_stream(block, datasource_manager)

        Observable.just(chunk).flat_map(blend_stream).subscribe(print)

        np.set_printoptions(threshold=np.nan)

        assert 3 ** len(chunk_index) * 7 * 3 == \
            datasource_manager.output_datasource_core.sum()
        # assert False

    def test_blend_multichannel_3d_cloudvolume(self, cloudvolume_factory):
        dataset_bounds = (slice(200, 207), slice(100, 107), slice(50, 57))
        task_size = (3, 3, 3)
        overlap = (1, 1, 1)
        cloud_volume_chunk_size = (2, 2, 2)
        voxel_offset = (200, 100, 50)
        input_data_type = 'uint8'
        output_data_type = 'float32'
        output_shape = (3,) + task_size

        block = Block(bounds=dataset_bounds, chunk_shape=task_size, overlap=overlap)
        assert block.num_chunks == (3, 3, 3)

        input_cloudvolume = cloudvolume_factory.create(
            'input', data_type=input_data_type, volume_size=block.shape, chunk_size=cloud_volume_chunk_size,
            voxel_offset=voxel_offset)
        output_cloudvolume_core = cloudvolume_factory.create(
            'output_core', data_type=output_data_type, volume_size=block.shape, chunk_size=cloud_volume_chunk_size,
            num_channels=3, voxel_offset=voxel_offset)
        output_cloudvolume_overlap = cloudvolume_factory.create(
            'output_overlap', data_type=output_data_type, volume_size=block.shape,
            chunk_size=cloud_volume_chunk_size, num_channels=3)

        repository = CloudVolumeDatasourceRepository(input_cloudvolume, output_cloudvolume_core,
                                                     output_cloudvolume_overlap)

        datasource_manager = DatasourceManager(repository)

        chunk_index = (1, 1, 1)

        chunk = block.unit_index_to_chunk(chunk_index)

        # set up test data
        datasource = datasource_manager.repository.get_datasource(chunk_index)
        datasource[chunk.slices] = np.ones(output_shape, dtype=np.dtype(output_data_type))
        for neighbor in UnitIterator().get_all_neighbors(chunk_index):
            datasource = datasource_manager.repository.get_datasource(neighbor)
            datasource[chunk.slices] = np.ones(output_shape, dtype=np.dtype(output_data_type))

        datasource_manager = DatasourceManager(repository)

        blend_stream = create_blend_stream(block, datasource_manager)

        Observable.just(chunk).flat_map(blend_stream).subscribe(print)

        assert 3 ** len(chunk_index) * 7 * 3 == \
            datasource_manager.output_datasource_core[dataset_bounds].sum()
