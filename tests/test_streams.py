import cProfile
import threading

import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from chunkblocks.models import Block
from rx import Observable

import chunkflow.streams as streams
from chunkflow.chunk_buffer import CacheMiss
from chunkflow.chunk_operations.blend_operation import AverageBlend
from chunkflow.chunk_operations.chunk_operation import ChunkOperation
from chunkflow.cloudvolume_datasource import create_buffered_cloudvolumeCZYX, create_sparse_overlap_cloudvolumeCZYX
from chunkflow.datasource_manager import DatasourceManager, OverlapRepository, SparseOverlapRepository
from chunkflow.streams import create_blend_stream, create_inference_and_blend_stream, create_preload_datasource_stream


class NumpyDatasourceManager(DatasourceManager):
    def __init__(self, output_shape, input_datasource=None, output_datasource=None, output_datasource_final=None,
                 *args, **kwargs):
        super().__init__(input_datasource=input_datasource,
                         output_datasource=output_datasource,
                         output_datasource_final=output_datasource_final,
                         overlap_repository=NumpyOverlapRepository(input_datasource, output_shape),
                         *args, **kwargs)

        # Hack for test to automatically create output datasources
        if self.output_datasource is None:
            self.output_datasource = self.overlap_repository.create(None)

        if self.output_datasource_final is None:
            self.output_datasource_final = self.overlap_repository.create(None)


class NumpyOverlapRepository(OverlapRepository):

    def __init__(self, input_datasource, output_shape):
        self.input_datasource = input_datasource
        self.output_shape = output_shape
        super().__init__()

    def create(self, mod_index, *args, **kwargs):
        offset = self.input_datasource.global_offset
        shape = self.input_datasource.shape

        if len(self.output_shape) > len(self.input_datasource.shape):
            extra_dimensions = len(self.output_shape) - len(self.input_datasource.shape)

            shape = self.output_shape[0:extra_dimensions] + shape
            offset = (0,) * extra_dimensions + offset

        return GlobalOffsetArray(np.zeros(shape), global_offset=offset)

    def clear(self, index):
        pass


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


class TestSubStreams:
    def test_input_stream(self, chunk_datasource_manager):
        task_shape = (10, 20, 20)
        overlap = (2, 5, 5)
        num_chunks = (3, 3, 3)

        input_datasource = chunk_datasource_manager.input_datasource
        offsets = input_datasource.voxel_offset[::-1]

        block = Block(num_chunks=num_chunks, offset=offsets, chunk_shape=task_shape, overlap=overlap)

        input_datasource[block.bounds] = np.ones(block.shape, input_datasource.dtype)

        input_stream = streams.create_input_stream(chunk_datasource_manager)
        chunk_datasource_manager.buffer_generator = create_buffered_cloudvolumeCZYX
        chunk_datasource_manager.overlap_repository = create_sparse_overlap_cloudvolumeCZYX(
            chunk_datasource_manager.output_datasource_final, block
        )

        from concurrent.futures import ProcessPoolExecutor
        chunk_datasource_manager.load_executor = ProcessPoolExecutor()
        chunk_datasource_manager.flush_executor = ProcessPoolExecutor()

        def validate():
            try:
                assert np.product(block.shape) == chunk_datasource_manager.get_buffer(
                    chunk_datasource_manager.input_datasource)[block.bounds].sum()
            except CacheMiss as cm:
                print('Misses are:', cm.misses)
                raise cm

        # Run with cold cache
        Observable.from_(block.chunk_iterator()).flat_map(input_stream).to_blocking().blocking_subscribe(print)
        validate()

        # Run with warm cache doesn't fail
        Observable.from_(block.chunk_iterator()).flat_map(input_stream).to_blocking().blocking_subscribe(print)
        validate()

        # Test with no buffer
        chunk_datasource_manager.buffer_generator = None
        chunk_datasource_manager.datasource_buffers.clear()
        Observable.from_(block.chunk_iterator()).flat_map(input_stream).to_blocking().blocking_subscribe(print)


class TestInferenceStream:

    def test_process_single_channel_2x2(self):
        bounds = (slice(0, 5), slice(0, 5))
        chunk_shape = (3, 3)
        output_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = NumpyDatasourceManager(input_datasource=fake_data, output_shape=output_shape)

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementInference(step=1),
            blend_operation=AverageBlend(block, weight_borders=False),
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
            datasource_manager.output_datasource.sum() + \
            datasource_manager.output_datasource_final.sum()

        test_chunk_1_1 = block.unit_index_to_chunk((1, 1))
        assert not block.is_checkpointed(test_chunk_1_1)
        Observable.just(test_chunk_1_1).flat_map(task_stream).subscribe(print)
        assert block.is_checkpointed(test_chunk_1_1)

        print(datasource_manager.output_datasource)
        print(datasource_manager.output_datasource_final)
        assert np.product(block.shape) == \
            datasource_manager.output_datasource.sum() + \
            datasource_manager.output_datasource_final.sum()

    def test_process_single_channel_3x3(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        output_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = NumpyDatasourceManager(input_datasource=fake_data, output_shape=output_shape)

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementInference(step=1),
            blend_operation=AverageBlend(block, weight_borders=False),
            datasource_manager=datasource_manager,
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).to_blocking().blocking_subscribe(print)

        assert np.product(block.shape) == \
            datasource_manager.output_datasource.sum() + \
            datasource_manager.output_datasource_final.sum()

    def test_process_multi_channel(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = NumpyDatasourceManager(input_datasource=fake_data, output_shape=(3,) + fake_data.shape)

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementThreeChannelInference(step=1),
            blend_operation=AverageBlend(block, weight_borders=False),
            datasource_manager=datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

        assert np.product(block.shape) * 111 == \
            datasource_manager.output_datasource.sum() + \
            datasource_manager.output_datasource_final.sum()

    def test_process_single_channel_3d(self):
        chunk_shape = (3, 6, 5)
        offset = (0, 0, 0)
        overlap = (1, 2, 2)
        num_chunks = (3, 3, 3)

        block = Block(num_chunks=num_chunks, offset=offset, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = NumpyDatasourceManager(input_datasource=fake_data, output_shape=fake_data.shape)

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementInference(step=1),
            blend_operation=AverageBlend(block, weight_borders=False),
            datasource_manager=datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

        np.set_printoptions(threshold=np.NaN, linewidth=200)
        print(datasource_manager.output_datasource)
        print(datasource_manager.output_datasource_final)

        assert np.product(block.shape) == \
            datasource_manager.output_datasource.sum() + \
            datasource_manager.output_datasource_final.sum()

    def test_process_multi_channel_3d(self):
        bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = NumpyDatasourceManager(input_datasource=fake_data, output_shape=(3,) + fake_data.shape)

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementThreeChannelInference(step=1),
            blend_operation=AverageBlend(block, weight_borders=False),
            datasource_manager=datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

        assert np.product(block.shape) * 111 == \
            datasource_manager.output_datasource.sum() + \
            datasource_manager.output_datasource_final.sum()

    def test_process_cloudvolume(self, chunk_datasource_manager):
        overlap = (2, 5, 5)
        num_chunks = (4, 3, 3)
        patch_shape = (4, 10, 10)

        input_datasource = chunk_datasource_manager.input_datasource
        offsets = input_datasource.voxel_offset[::-1]

        block = Block(num_chunks=num_chunks, offset=offsets, chunk_shape=patch_shape, overlap=overlap)

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementThreeChannelInference(step=1, output_dtype=np.dtype(
                chunk_datasource_manager.output_datasource.dtype)),
            blend_operation=AverageBlend(block, weight_borders=False),
            datasource_manager=chunk_datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

        print('block shape is ', block.shape)
        np.set_printoptions(threshold=np.NaN, linewidth=200)
        print(chunk_datasource_manager.output_datasource[block.bounds])
        print(chunk_datasource_manager.output_datasource_final[block.bounds])

        assert np.product(block.shape) * 111 == \
            chunk_datasource_manager.output_datasource[block.bounds].sum() + \
            chunk_datasource_manager.output_datasource_final[block.bounds].sum()

    def test_process_cloudvolume_buffer(self, chunk_datasource_manager):
        overlap = (2, 5, 5)
        num_chunks = (4, 3, 3)
        patch_shape = (4, 10, 10)

        input_datasource = chunk_datasource_manager.input_datasource
        offsets = input_datasource.voxel_offset[::-1]

        block = Block(num_chunks=num_chunks, offset=offsets, chunk_shape=patch_shape, overlap=overlap)

        chunk_datasource_manager.buffer_generator = create_buffered_cloudvolumeCZYX

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementThreeChannelInference(step=1, output_dtype=np.dtype(
                chunk_datasource_manager.output_datasource.dtype)),
            blend_operation=AverageBlend(block, weight_borders=False),
            datasource_manager=chunk_datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

        assert np.product(block.shape) * 111 == \
            chunk_datasource_manager.output_datasource[block.bounds].sum() + \
            chunk_datasource_manager.output_datasource_final[block.bounds].sum()

    def test_process_cloudvolume_sparse_buffered(self, chunk_datasource_manager):
        task_shape = (5, 10, 10)
        overlap = (1, 2, 2)
        num_chunks = (3, 3, 3)

        input_datasource = chunk_datasource_manager.input_datasource
        offsets = input_datasource.voxel_offset[::-1]
        bounds = tuple(slice(o, o + (ts - olap) * nc + olap) for o, ts, olap, nc in zip(offsets, task_shape, overlap,
                                                                                        num_chunks))

        block = Block(num_chunks=num_chunks, offset=offsets, chunk_shape=task_shape, overlap=overlap)

        chunk_datasource_manager.buffer_generator = create_buffered_cloudvolumeCZYX
        chunk_datasource_manager.overlap_repository = create_sparse_overlap_cloudvolumeCZYX(
            chunk_datasource_manager.output_datasource_final, block
        )
        from concurrent.futures import ProcessPoolExecutor
        chunk_datasource_manager.load_executor = ProcessPoolExecutor()
        chunk_datasource_manager.flush_executor = ProcessPoolExecutor()

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementThreeChannelInference(step=1, output_dtype=np.dtype(
                chunk_datasource_manager.output_datasource.dtype)),
            blend_operation=AverageBlend(block, weight_borders=False),
            datasource_manager=chunk_datasource_manager
        )

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).to_blocking().blocking_subscribe(print)

        np.set_printoptions(threshold=np.NaN, linewidth=400)
        print(chunk_datasource_manager.output_datasource[bounds])
        print(chunk_datasource_manager.output_datasource_final[bounds])
        assert np.product(block.shape) * 111 == \
            chunk_datasource_manager.output_datasource[bounds].sum() + \
            chunk_datasource_manager.output_datasource_final[bounds].sum()


class TestBlendStream:

    def test_blend(self):
        dataset_bounds = (slice(0, 7), slice(0, 7))
        task_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=dataset_bounds, chunk_shape=task_shape, overlap=overlap)
        assert block.num_chunks == (3, 3)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))

        datasource_manager = NumpyDatasourceManager(input_datasource=fake_data, output_shape=fake_data.shape)

        chunk_index = (1, 1)

        chunk = block.unit_index_to_chunk(chunk_index)

        # set up test data
        datasource_manager.create_overlap_datasources(chunk_index)
        for datasource in datasource_manager.overlap_repository.datasources.values():
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

        datasource_manager = NumpyDatasourceManager(input_datasource=fake_data, output_shape=fake_data.shape)

        chunk_index = (1, 1, 1)

        chunk = block.unit_index_to_chunk(chunk_index)

        # set up test data
        datasource_manager.create_overlap_datasources(chunk_index)
        for datasource in datasource_manager.overlap_repository.datasources.values():
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

        block_datasource_manager = NumpyDatasourceManager(input_datasource=fake_data, output_shape=output_shape)

        for chunk in block.chunk_iterator():
            overlap_datasource = block_datasource_manager.overlap_repository.get_datasource(chunk.unit_index)
            core_datasource = block_datasource_manager.output_datasource

            core_slices = (slice(None, None),) + chunk.core_slices()
            core_datasource[core_slices] = np.ones(core_datasource[core_slices].shape,
                                                   dtype=np.dtype(core_datasource.dtype))

            for overlap_slices in chunk.border_slices():
                overlap_slices = (slice(None, None),) + overlap_slices
                overlap_datasource[overlap_slices] = np.ones(overlap_datasource[overlap_slices].shape,
                                                             dtype=overlap_datasource.dtype)

        # set up test data

        blend_stream = create_blend_stream(block, block_datasource_manager)

        Observable.from_(block.chunk_iterator()).flat_map(blend_stream).subscribe(print)

        assert np.product(task_shape) * np.product(block.num_chunks) * 3 == \
            block_datasource_manager.output_datasource.sum()

    def test_blend_multichannel_3d_cloudvolume(self, block_datasource_manager):
        task_shape = (10, 20, 20)
        overlap = (2, 5, 5)
        num_chunks = (3, 3, 3)

        input_datasource = block_datasource_manager.input_datasource
        offsets = input_datasource.voxel_offset[::-1]

        block = Block(num_chunks=num_chunks, offset=offsets, chunk_shape=task_shape, overlap=overlap)

        # set up test data
        for chunk in block.chunk_iterator():
            overlap_datasource = block_datasource_manager.overlap_repository.get_datasource(chunk.unit_index)
            core_datasource = block_datasource_manager.output_datasource

            core_slices = chunk.core_slices()
            core_datasource[core_slices] = np.ones((3,) + tuple(s.stop - s.start for s in core_slices),
                                                   dtype=np.dtype(core_datasource.data_type))

            for overlap_slices in chunk.border_slices():
                overlap_datasource[overlap_slices] = np.ones((3,) + tuple(s.stop - s.start for s in overlap_slices),
                                                             dtype=overlap_datasource.data_type)

        blend_stream = create_blend_stream(block, block_datasource_manager)
        Observable.from_(block.chunk_iterator()).flat_map(blend_stream).subscribe(print)

        assert np.product(task_shape) * np.product(block.num_chunks) * 3 == \
            block_datasource_manager.output_datasource[block.bounds].sum()

    def test_blend_multichannel_3d_cloudvolume_buffered(self, block_datasource_manager):
        task_shape = (10, 20, 20)
        overlap = (2, 5, 5)
        num_chunks = (3, 3, 3)

        input_datasource = block_datasource_manager.input_datasource
        offsets = input_datasource.voxel_offset[::-1]

        block = Block(num_chunks=num_chunks, offset=offsets, chunk_shape=task_shape, overlap=overlap)
        block_datasource_manager.buffer_generator = create_buffered_cloudvolumeCZYX

        # set up test data
        data_sum = 0
        for chunk in list(block.chunk_iterator()):
            overlap_datasource = block_datasource_manager.overlap_repository.get_datasource(chunk.unit_index)
            core_datasource = block_datasource_manager.output_datasource

            core_slices = chunk.core_slices()
            core_datasource[core_slices] = np.ones((3,) + tuple(s.stop - s.start for s in core_slices),
                                                   dtype=np.dtype(core_datasource.data_type)) * 3
            data_sum += core_datasource[core_slices].sum()

            for overlap_slices in chunk.border_slices():
                overlap_datasource[overlap_slices] = np.ones((3,) + tuple(s.stop - s.start for s in overlap_slices),
                                                             dtype=overlap_datasource.data_type)
                data_sum += overlap_datasource[overlap_slices].sum()

        blend_stream = create_blend_stream(block, block_datasource_manager)
        (
            Observable.from_(list(block.chunk_iterator()))
            .flat_map(create_preload_datasource_stream(block, block_datasource_manager,
                                                       block_datasource_manager.output_datasource))
            .flat_map(blend_stream)
            .reduce(lambda x, y: x)
            .map(lambda _: block_datasource_manager.flush(None, datasource=block_datasource_manager.output_datasource))
            .flat_map(lambda chunk_or_future: Observable.from_item_or_future(chunk_or_future))
            .subscribe(print)
        )
        np.set_printoptions(threshold=np.nan, linewidth=400)
        print(block_datasource_manager.output_datasource[block.bounds][0])
        assert data_sum == block_datasource_manager.output_datasource[block.bounds].sum()


class TestPerformance:

    def test_performance(self, chunk_datasource_manager):
        """
        Use this to benchmark inference performance
        pytest  --capture=no --basetemp=/tmp/ramdisk/folder tests/test_streams.py::TestPerformance
        """
        should_profile = False
        profile_file = '/home/wwong/src/chunkflow/prof-2-notime.cprof'

        # change num_chunks when benchmarking!
        # benchmark with:
        #    conftest.py volume_size = (600, 4096, 4096) and CLOUD_VOLUME_CHUNK_SIZE = (12, 96, 96)
        #    here patch_shape = (16, 128, 128) and  overlap = (4, 32, 32)
        num_chunks = (2, 2, 2)
        patch_shape = (8, 16, 16)
        overlap = (2, 4, 4)
        offset = (200, 100, 50)

        dtype = chunk_datasource_manager.output_datasource.dtype

        block = Block(offset=offset, num_chunks=num_chunks, chunk_shape=patch_shape, overlap=overlap)

        datasource_manager = DatasourceManager(
            input_datasource=chunk_datasource_manager.input_datasource,
            output_datasource=chunk_datasource_manager.output_datasource,
            output_datasource_final=chunk_datasource_manager.output_datasource_final,
            overlap_repository=SparseOverlapRepository(
                block=block,
                channel_dimensions=(3,),
                dtype=chunk_datasource_manager.output_datasource.dtype,
            )
        )

        chunk_datasource_manager.create_overlap_datasources(patch_shape)
        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementThreeChannelInference(step=1, output_dtype=dtype),
            blend_operation=AverageBlend(block, weight_borders=False),
            datasource_manager=datasource_manager,
        )
        import time

        stats = dict()
        stats['completed'] = 0
        stats['start'] = time.time()
        stats['previous_time'] = stats['start']
        stats['running_sum'] = 0.0

        lock = threading.Lock()

        def on_subscribe(item):
            lock.acquire()
            now = time.time()
            stats['completed'] += 1
            if (stats['completed'] == 0):
                return True
            elapsed = now - stats['previous_time']
            stats['previous_time'] = now
            stats['running_sum'] += elapsed
            print('Average:', stats['running_sum']/stats['completed'], '\t', stats['completed'])
            lock.release()
            return True

        if should_profile:
            profile = cProfile.Profile()

        import traceback

        def on_error(error):
            print('\n\n\n\nerror error *&)*&*&)*\n\n')
            self.error = error
            traceback.print_exception(None, error, error.__traceback__)
            raise error

        if should_profile:
            profile.enable()

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).to_blocking().blocking_subscribe(
            on_subscribe, on_error=on_error)

        print('\n\n\ncompleted ', len(list(block.chunk_iterator())), ' chunks in ', time.time() - stats['start'])

        if should_profile:
            profile.disable()
            profile.dump_stats(profile_file)

        actual = datasource_manager.output_datasource[block.bounds].sum() + \
            datasource_manager.output_datasource_final[block.bounds].sum()
        assert np.product(block.shape) * 111 == actual


class TestStreamIntegration:

    def test_stream_inference_then_blend(self, chunk_datasource_manager, block_datasource_manager):
        from chunkflow.datasource_manager import get_absolute_index

        input_datasource = chunk_datasource_manager.input_datasource
        offset = input_datasource.voxel_offset[::-1]
        overlap = (1, 4, 4)
        patch_shape = (5, 10, 10)
        num_patches_per_task = (2, 2, 2)
        task_shape = tuple((ps - olap) * num + olap for ps, olap, num in zip(patch_shape, overlap,
                                                                             num_patches_per_task))
        dataset_block = Block(offset=offset, num_chunks=[3, 3, 3], chunk_shape=task_shape, overlap=overlap)

        np.set_printoptions(threshold=np.NaN, linewidth=400)
        for chunk in list(dataset_block.chunk_iterator()):
            # get the corresponding overlap index for this
            absolute_index = get_absolute_index(chunk.offset, overlap, task_shape)
            output_cloudvolume_overlap = block_datasource_manager.overlap_repository.get_datasource(absolute_index)
            chunk_datasource_manager.output_datasource = output_cloudvolume_overlap

            task_block = Block(num_chunks=num_patches_per_task, offset=chunk.offset, chunk_shape=patch_shape,
                               overlap=overlap)

            chunk_datasource_manager.buffer_generator = create_buffered_cloudvolumeCZYX
            chunk_datasource_manager.overlap_repository = create_sparse_overlap_cloudvolumeCZYX(
                chunk_datasource_manager.output_datasource_final, task_block
            )

            task_stream = create_inference_and_blend_stream(
                block=task_block,
                inference_operation=IncrementThreeChannelInference(step=1, output_dtype=np.dtype(
                    chunk_datasource_manager.output_datasource.dtype)),
                blend_operation=AverageBlend(task_block),
                datasource_manager=chunk_datasource_manager
            )

            Observable.from_(task_block.chunk_iterator()).flat_map(task_stream).to_blocking().blocking_subscribe(print)

        block_datasource_manager.buffer_generator = create_buffered_cloudvolumeCZYX

        blend_stream = create_blend_stream(dataset_block, block_datasource_manager)
        (
            Observable.from_(list(dataset_block.chunk_iterator()))
            .flat_map(create_preload_datasource_stream(dataset_block, block_datasource_manager,
                                                       block_datasource_manager.output_datasource))
            .flat_map(blend_stream)
            .reduce(lambda x, y: x)
            .map(lambda _: block_datasource_manager.flush(None, datasource=block_datasource_manager.output_datasource))
            .flat_map(lambda chunk_or_future: Observable.from_item_or_future(chunk_or_future))
            .subscribe(print)
        )

        inner_bounds = tuple(slice(s.start + o, s.stop - o) for s, o in zip(dataset_block.bounds, overlap))
        inner_shape = tuple(sh - o * 2 for sh, o in zip(dataset_block.shape, overlap))
        print(dataset_block.bounds, inner_bounds)
        print(task_block.shape, inner_shape)

        assert np.product(inner_shape) * 111 == \
            chunk_datasource_manager.output_datasource_final[inner_bounds].sum()


class TestRxExtension:

    def test_distinct_hash(self):
        completed_distinct = []
        completed_hash = []

        num_values = 10
        # regular distinct does not work
        (
            Observable.from_(range(num_values))
            .flat_map(lambda i: Observable.just(i).flat_map(lambda ii: [ii, ii + 1]).distinct())
            .subscribe(lambda i: completed_distinct.append(i))
        )
        assert len(completed_distinct) != len(set(completed_distinct))

        hashset = set()
        (
            Observable.from_(range(num_values))
            .flat_map(lambda i: Observable.just(i).flat_map(lambda ii: [ii, ii + 1]).distinct_hash(seed=hashset))
            .subscribe(lambda i: completed_hash.append(i))
        )
        assert len(completed_hash) == len(set(completed_hash))
