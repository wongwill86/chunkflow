import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from chunkblocks.models import Block
from rx import Observable

from chunkflow.chunk_operations.blend_operation import AverageBlend
from chunkflow.chunk_operations.chunk_operation import ChunkOperation
from chunkflow.datasource_manager import DatasourceManager, DatasourceRepository
from chunkflow.streams import create_blend_stream, create_inference_and_blend_stream


class NumpyDatasource(DatasourceRepository):
    def __init__(self, output_shape,
                 output_datasource=None, output_datasource_overlap=None, *args, **kwargs):
        self.output_shape = output_shape
        super().__init__(output_datasource=output_datasource,
                         output_datasource_overlap=output_datasource_overlap,
                         *args, **kwargs)

        if self.output_datasource is None:
            self.output_datasource = self.create(None)

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
            datasource_manager.repository.output_datasource.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum()

        test_chunk_1_1 = block.unit_index_to_chunk((1, 1))
        assert not block.is_checkpointed(test_chunk_1_1)
        Observable.just(test_chunk_1_1).flat_map(task_stream).subscribe(print)
        assert block.is_checkpointed(test_chunk_1_1)

        assert np.product(block.shape) == \
            datasource_manager.repository.output_datasource.sum() + \
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
            datasource_manager.repository.output_datasource.sum() + \
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
            datasource_manager.repository.output_datasource.sum() + \
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
            datasource_manager.repository.output_datasource.sum() + \
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
            datasource_manager.repository.output_datasource.sum() + \
            datasource_manager.repository.output_datasource_overlap.sum()

    def test_process_cloudvolume(self, cloudvolume_datasource_manager):
        bounds = (slice(200, 203), slice(100, 103), slice(50, 53))
        chunk_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementThreeChannelInference(step=1, output_dtype=np.dtype(
                cloudvolume_datasource_manager.output_datasource.data_type)),
            blend_operation=AverageBlend(block),
            datasource_manager=cloudvolume_datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

        assert np.product(block.shape) * 111 == \
            cloudvolume_datasource_manager.repository.output_datasource[bounds].sum() + \
            cloudvolume_datasource_manager.repository.output_datasource_overlap[bounds].sum()


class aTestBlendStream:

    def test_blend(self):
        dataset_bounds = (slice(0, 7), slice(0, 7))
        task_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=dataset_bounds, chunk_shape=task_shape, overlap=overlap)
        assert block.num_chunks == (3, 3)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))

        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=fake_data.shape))

        chunk_index = (1, 1)

        chunk = block.unit_index_to_chunk(chunk_index)

        # set up test data
        datasource_manager.repository.create_intermediate_datasources(chunk_index)
        for datasource in datasource_manager.repository.intermediate_datasources.values():
            datasource[chunk.slices] = 1

        blend_stream = create_blend_stream(block, datasource_manager)

        Observable.just(chunk).flat_map(blend_stream).subscribe(print)

        assert 3 ** len(chunk_index) * 3 == \
            datasource_manager.output_datasource.sum()

    def test_blend_3d(self):
        dataset_bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        task_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=dataset_bounds, chunk_shape=task_shape, overlap=overlap)
        assert block.num_chunks == (3, 3, 3)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))

        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=fake_data.shape))

        chunk_index = (1, 1, 1)

        chunk = block.unit_index_to_chunk(chunk_index)

        # set up test data
        datasource_manager.repository.create_intermediate_datasources(chunk_index)
        for datasource in datasource_manager.repository.intermediate_datasources.values():
            datasource[chunk.slices] = 1

        blend_stream = create_blend_stream(block, datasource_manager)

        Observable.just(chunk).flat_map(blend_stream).subscribe(print)

        assert 3 ** len(chunk_index) * 7 == \
            datasource_manager.output_datasource.sum()

    def test_blend_multichannel_3d(self):
        dataset_bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        task_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=dataset_bounds, chunk_shape=task_shape, overlap=overlap)
        assert block.num_chunks == (3, 3, 3)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        output_shape = (3,) + fake_data.shape

        datasource_manager = DatasourceManager(
            NumpyDatasource(input_datasource=fake_data, output_shape=output_shape))

        chunk_index = (1, 1, 1)

        chunk = block.unit_index_to_chunk(chunk_index)

        # set up test data
        datasource_manager.repository.create_intermediate_datasources(chunk_index)
        for datasource in datasource_manager.repository.intermediate_datasources.values():
            datasource[(slice(None),) + chunk.slices] = 1

        blend_stream = create_blend_stream(block, datasource_manager)

        Observable.just(chunk).flat_map(blend_stream).subscribe(print)

        np.set_printoptions(threshold=np.nan)

        assert 3 ** len(chunk_index) * 7 * 3 == \
            datasource_manager.output_datasource.sum()
        # assert False

    def test_blend_multichannel_3d_cloudvolume(self, cloudvolume_datasource_manager):
        task_shape = (3, 3, 3)
        overlap = (1, 1, 1)
        output_shape = (3,) + task_shape

        input_datasource = cloudvolume_datasource_manager.repository.input_datasource
        offsets = input_datasource.voxel_offset[::-1]
        volume_size = input_datasource.volume_size[::-1]

        dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offsets, volume_size))

        block = Block(bounds=dataset_bounds, chunk_shape=task_shape, overlap=overlap)
        # assert block.num_chunks == (3, 3, 3)

        chunk_index = (1, 1, 1)

        chunk = block.unit_index_to_chunk(chunk_index)

        # set up test data
        cloudvolume_datasource_manager.repository.create_intermediate_datasources(chunk_index)
        for datasource in cloudvolume_datasource_manager.repository.intermediate_datasources.values():
            datasource[chunk.slices] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

        datasource = cloudvolume_datasource_manager.repository.get_datasource(chunk_index)
        datasource[chunk.slices] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

        blend_stream = create_blend_stream(block, cloudvolume_datasource_manager)

        Observable.just(chunk).flat_map(blend_stream).subscribe(print)

        assert 3 ** len(chunk_index) * 7 * 3 == \
            cloudvolume_datasource_manager.output_datasource[dataset_bounds].sum()
