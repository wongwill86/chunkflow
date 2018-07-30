import cProfile
import multiprocessing

import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from chunkblocks.models import Block
from rx import Observable
from rx.concurrency import ThreadPoolScheduler

from chunkflow.chunk_operations.blend_operation import AverageBlend
from chunkflow.chunk_operations.chunk_operation import ChunkOperation
from chunkflow.datasource_manager import DatasourceManager, DatasourceRepository
from chunkflow.streams import create_blend_stream, create_inference_and_blend_stream


class NumpyDatasourceRepository(DatasourceRepository):
    def __init__(self, output_shape,
                 output_datasource=None, output_datasource_final=None, *args, **kwargs):
        self.output_shape = output_shape
        super().__init__(output_datasource=output_datasource,
                         output_datasource_final=output_datasource_final,
                         *args, **kwargs)

        if self.output_datasource is None:
            self.output_datasource = self.create(None)

        if self.output_datasource_final is None:
            self.output_datasource_final = self.create(None)

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
        print('run inference on chunk', chunk.unit_index)
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

    # def test_process_single_channel_2x2(self):
    #     bounds = (slice(0, 5), slice(0, 5))
    #     chunk_shape = (3, 3)
    #     output_shape = (3, 3)
    #     overlap = (1, 1)

    #     block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

    #     fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
    #     datasource_manager = DatasourceManager(
    #         NumpyDatasourceRepository(input_datasource=fake_data, output_shape=output_shape))

    #     task_stream = create_inference_and_blend_stream(
    #         block=block,
    #         inference_operation=IncrementInference(step=1),
    #         blend_operation=AverageBlend(block),
    #         datasource_manager=datasource_manager
    #     )

    #     test_chunk_0_0 = block.unit_index_to_chunk((0, 0))
    #     assert not block.is_checkpointed(test_chunk_0_0)
    #     Observable.just(test_chunk_0_0).flat_map(task_stream).subscribe(print)
    #     assert block.is_checkpointed(test_chunk_0_0)

    #     test_chunk_0_1 = block.unit_index_to_chunk((0, 1))
    #     assert not block.is_checkpointed(test_chunk_0_1)
    #     Observable.just(test_chunk_0_1).flat_map(task_stream).subscribe(print)
    #     assert block.is_checkpointed(test_chunk_0_1)

    #     test_chunk_1_0 = block.unit_index_to_chunk((1, 0))
    #     assert not block.is_checkpointed(test_chunk_1_0)
    #     Observable.just(test_chunk_1_0).flat_map(task_stream).subscribe(print)
    #     assert block.is_checkpointed(test_chunk_1_0)

    #     assert 0 == \
    #         datasource_manager.repository.output_datasource.sum() + \
    #         datasource_manager.repository.output_datasource_final.sum()

    #     test_chunk_1_1 = block.unit_index_to_chunk((1, 1))
    #     assert not block.is_checkpointed(test_chunk_1_1)
    #     Observable.just(test_chunk_1_1).flat_map(task_stream).subscribe(print)
    #     assert block.is_checkpointed(test_chunk_1_1)

    #     print(datasource_manager.repository.output_datasource)
    #     print(datasource_manager.repository.output_datasource_final)
    #     assert np.product(block.shape) == \
    #         datasource_manager.repository.output_datasource.sum() + \
    #         datasource_manager.repository.output_datasource_final.sum()

    def test_process_single_channel_3x3(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        output_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)
        optimal_thread_count = multiprocessing.cpu_count()
        scheduler = ThreadPoolScheduler(optimal_thread_count)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
        datasource_manager = DatasourceManager(
            NumpyDatasourceRepository(input_datasource=fake_data, output_shape=output_shape))

        task_stream = create_inference_and_blend_stream(
            block=block,
            inference_operation=IncrementInference(step=1),
            blend_operation=AverageBlend(block),
            datasource_manager=datasource_manager,
            scheduler=scheduler
        )
        def do_error(error):
            raise error

        Observable.from_(block.chunk_iterator()).flat_map(task_stream).to_blocking().blocking_subscribe(
            print, on_error=do_error)

        print(datasource_manager.repository.output_datasource)
        print(datasource_manager.repository.output_datasource_final)
        assert np.product(block.shape) == \
            datasource_manager.repository.output_datasource.sum() + \
            datasource_manager.repository.output_datasource_final.sum()
        assert False

    # def test_process_multi_channel(self):
    #     bounds = (slice(0, 7), slice(0, 7))
    #     chunk_shape = (3, 3)
    #     overlap = (1, 1)

    #     block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

    #     fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
    #     datasource_manager = DatasourceManager(
    #         NumpyDatasourceRepository(input_datasource=fake_data, output_shape=(3,) + fake_data.shape))

    #     task_stream = create_inference_and_blend_stream(
    #         block=block,
    #         inference_operation=IncrementThreeChannelInference(step=1),
    #         blend_operation=AverageBlend(block),
    #         datasource_manager=datasource_manager
    #     )

    #     Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

    #     assert np.product(block.shape) * 111 == \
    #         datasource_manager.repository.output_datasource.sum() + \
    #         datasource_manager.repository.output_datasource_final.sum()

#     def test_process_single_channel_3d(self):
#         bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
#         chunk_shape = (3, 3, 3)
#         overlap = (1, 1, 1)

#         block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

#         fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
#         datasource_manager = DatasourceManager(
#             NumpyDatasourceRepository(input_datasource=fake_data, output_shape=fake_data.shape))

#         task_stream = create_inference_and_blend_stream(
#             block=block,
#             inference_operation=IncrementInference(step=1),
#             blend_operation=AverageBlend(block),
#             datasource_manager=datasource_manager
#         )

#         Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

#         assert np.product(block.shape) == \
#             datasource_manager.repository.output_datasource.sum() + \
#             datasource_manager.repository.output_datasource_final.sum()

#     def test_process_multi_channel_3d(self):
#         bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
#         chunk_shape = (3, 3, 3)
#         overlap = (1, 1, 1)

#         block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

#         fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
#         datasource_manager = DatasourceManager(
#             NumpyDatasourceRepository(input_datasource=fake_data, output_shape=(3,) + fake_data.shape))

#         task_stream = create_inference_and_blend_stream(
#             block=block,
#             inference_operation=IncrementThreeChannelInference(step=1),
#             blend_operation=AverageBlend(block),
#             datasource_manager=datasource_manager
#         )

#         Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

#         assert np.product(block.shape) * 111 == \
#             datasource_manager.repository.output_datasource.sum() + \
#             datasource_manager.repository.output_datasource_final.sum()

#     def test_process_cloudvolume(self, chunk_datasource_manager):
#         bounds = (slice(200, 203), slice(100, 103), slice(50, 53))
#         chunk_shape = (3, 3, 3)
#         overlap = (1, 1, 1)

#         block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

#         task_stream = create_inference_and_blend_stream(
#             block=block,
#             inference_operation=IncrementThreeChannelInference(step=1, output_dtype=np.dtype(
#                 chunk_datasource_manager.output_datasource.data_type)),
#             blend_operation=AverageBlend(block),
#             datasource_manager=chunk_datasource_manager
#         )

#         Observable.from_(block.chunk_iterator()).flat_map(task_stream).subscribe(print)

#         print(chunk_datasource_manager.repository.output_datasource[bounds])
#         print(chunk_datasource_manager.repository.output_datasource_final[bounds])
#         assert np.product(block.shape) * 111 == \
#             chunk_datasource_manager.repository.output_datasource[bounds].sum() + \
#             chunk_datasource_manager.repository.output_datasource_final[bounds].sum()


# class TestBlendStream:

#     def test_blend(self):
#         dataset_bounds = (slice(0, 7), slice(0, 7))
#         task_shape = (3, 3)
#         overlap = (1, 1)

#         block = Block(bounds=dataset_bounds, chunk_shape=task_shape, overlap=overlap)
#         assert block.num_chunks == (3, 3)

#         fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))

#         datasource_manager = DatasourceManager(
#             NumpyDatasourceRepository(input_datasource=fake_data, output_shape=fake_data.shape))

#         chunk_index = (1, 1)

#         chunk = block.unit_index_to_chunk(chunk_index)

#         # set up test data
#         datasource_manager.repository.create_overlap_datasources(chunk_index)
#         for datasource in datasource_manager.repository.overlap_datasources.values():
#             datasource[chunk.slices] = 1

#         blend_stream = create_blend_stream(block, datasource_manager)

#         Observable.just(chunk).flat_map(blend_stream).subscribe(print)

#         assert 3 ** len(chunk_index) * 3 == \
#             datasource_manager.output_datasource.sum()

#     def test_blend_3d(self):
#         dataset_bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
#         task_shape = (3, 3, 3)
#         overlap = (1, 1, 1)

#         block = Block(bounds=dataset_bounds, chunk_shape=task_shape, overlap=overlap)
#         assert block.num_chunks == (3, 3, 3)

#         fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))

#         datasource_manager = DatasourceManager(
#             NumpyDatasourceRepository(input_datasource=fake_data, output_shape=fake_data.shape))

#         chunk_index = (1, 1, 1)

#         chunk = block.unit_index_to_chunk(chunk_index)

#         # set up test data
#         datasource_manager.repository.create_overlap_datasources(chunk_index)
#         for datasource in datasource_manager.repository.overlap_datasources.values():
#             datasource[chunk.slices] = 1

#         blend_stream = create_blend_stream(block, datasource_manager)

#         Observable.just(chunk).flat_map(blend_stream).subscribe(print)

#         assert 3 ** len(chunk_index) * 7 == \
#             datasource_manager.output_datasource.sum()

#     def test_blend_multichannel_3d(self):
#         dataset_bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
#         task_shape = (3, 3, 3)
#         overlap = (1, 1, 1)

#         block = Block(bounds=dataset_bounds, chunk_shape=task_shape, overlap=overlap)
#         assert block.num_chunks == (3, 3, 3)

#         fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0,) * len(block.shape))
#         output_shape = (3,) + fake_data.shape

#         datasource_manager = DatasourceManager(
#             NumpyDatasourceRepository(input_datasource=fake_data, output_shape=output_shape))

#         chunk_index = (1, 1, 1)

#         chunk = block.unit_index_to_chunk(chunk_index)

#         # set up test data
#         datasource_manager.repository.create_overlap_datasources(chunk_index)
#         for datasource in datasource_manager.repository.overlap_datasources.values():
#             datasource[(slice(None),) + chunk.slices] = 1

#         blend_stream = create_blend_stream(block, datasource_manager)

#         Observable.just(chunk).flat_map(blend_stream).subscribe(print)

#         np.set_printoptions(threshold=np.nan)

#         assert 3 ** len(chunk_index) * 7 * 3 == \
#             datasource_manager.output_datasource.sum()
#         # assert False

#     def test_blend_multichannel_3d_cloudvolume(self, block_datasource_manager):
#         task_shape = (3, 30, 30)
#         overlap = (1, 10, 10)
#         output_shape = (3,) + task_shape

#         input_datasource = block_datasource_manager.repository.input_datasource
#         offsets = input_datasource.voxel_offset[::-1]
#         volume_size = input_datasource.volume_size[::-1]

#         dataset_bounds = tuple(slice(o, o + s) for o, s in zip(offsets, volume_size))

#         block = Block(bounds=dataset_bounds, chunk_shape=task_shape, overlap=overlap)
#         # assert block.num_chunks == (3, 3, 3)

#         chunk_index = (1, 1, 1)

#         chunk = block.unit_index_to_chunk(chunk_index)

#         # set up test data
#         block_datasource_manager.repository.create_overlap_datasources(chunk_index)
#         for datasource in block_datasource_manager.repository.overlap_datasources.values():
#             datasource[chunk.slices] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

#         datasource = block_datasource_manager.repository.get_datasource(chunk_index)
#         datasource[chunk.slices] = np.ones(output_shape, dtype=np.dtype(datasource.data_type))

#         blend_stream = create_blend_stream(block, block_datasource_manager)

#         Observable.just(chunk).flat_map(blend_stream).subscribe(print)

#         assert np.product(task_shape) * 7 * 3 == \
#             block_datasource_manager.output_datasource[dataset_bounds].sum()


# class TestPerformance:

#     def test_performance(self, chunk_datasource_manager):
#         """
#         Use this to benchmark inference performance
#         pytest  --capture=no --basetemp=/tmp/ramdisk/folder tests/test_streams.py::TestPerformance
#         """
#         should_profile = False
#         profile_file = '/home/wwong/src/chunkflow/prof-2-notime.cprof'
#         optimal_thread_count = multiprocessing.cpu_count()
#         scheduler = ThreadPoolScheduler(optimal_thread_count)

#         # change num_chunks for real test!
#         num_chunks = (2, 2, 2)
#         patch_shape = (16, 128, 128)
#         overlap = (4, 32, 32)
#         offset = (200, 100, 50)

#         block = Block(offset=offset, num_chunks=num_chunks, chunk_shape=patch_shape, overlap=overlap)

#         fake_data = GlobalOffsetArray(np.zeros(block.shape, dtype=np.float32), global_offset=offset)
#         datasource_manager = DatasourceManager(
#             NumpyDatasourceRepository(input_datasource=fake_data, output_shape=(3,) + fake_data.shape))

#         chunk_datasource_manager.repository.create_overlap_datasources(patch_shape)
#         task_stream = create_inference_and_blend_stream(
#             block=block,
#             inference_operation=IncrementThreeChannelInference(step=1, output_dtype=np.float32),
#             blend_operation=AverageBlend(block),
#             datasource_manager=datasource_manager,
#             scheduler=scheduler
#         )
#         import time

#         stats = dict()
#         stats['completed'] = 0
#         stats['start'] = time.time()
#         stats['previous_time'] = stats['start']
#         stats['running_sum'] = 0.0
#         import threading

#         lock = threading.Lock()

#         def on_subscribe(item):
#             lock.acquire()
#             now = time.time()
#             stats['completed'] += 1
#             if (stats['completed'] == 0):
#                 return True
#             elapsed = now - stats['previous_time']
#             stats['previous_time'] = now
#             stats['running_sum'] += elapsed
#             print(stats['running_sum']/stats['completed'], '\t', stats['completed'])
#             lock.release()
#             return True

#         if should_profile:
#             profile = cProfile.Profile()

#         import traceback

#         def on_error(error):
#             print('\n\n\n\nerror error *&)*&*&)*\n\n')
#             self.error = error
#             traceback.print_exception(None, error, error.__traceback__)
#             # raise error

#         if should_profile:
#             profile.enable()

#         Observable.from_(block.chunk_iterator()).flat_map(task_stream).to_blocking().blocking_subscribe(
#             on_subscribe, on_error=on_error, timeout=5)

#         print('completed ', len(list(block.chunk_iterator())), ' chunks in ', time.time() - stats['start'])

#         if should_profile:
#             profile.disable()
#             profile.dump_stats(profile_file)

#         actual = datasource_manager.repository.output_datasource.sum() + \
#             datasource_manager.repository.output_datasource_final.sum()
#         assert np.product(block.shape) * 111 == actual
