import traceback
from collections import deque
from datetime import datetime
from functools import reduce
from threading import current_thread
from chunkblocks.iterators import UnitIterator
from chunkflow.streams import blocking_subscribe
import itertools
import functools
from memory_profiler import profile
from concurrent.futures import as_completed, ProcessPoolExecutor
import psutil
import time
import os
import gc
import numpy as np

from rx import Observable


class ReadyNeighborIterator(UnitIterator):
    def __init__(self, num_chunks):
        self.num_chunks = num_chunks

    def generate_queue(self, start):
        """
        Ghetto space filling curve that's kinda like z-order, but probably worse and not performant
        """
        print('generating queue')
        unfinished = np.ones(self.num_chunks, dtype=np.uint8)
        it = np.nditer(unfinished, flags=['multi_index'])
        while not it.finished:
            unfinished[it.multi_index] += len(self.get_all_neighbors(it.multi_index, self.num_chunks))
            it.iternext()

        # print(unfinished, np.any(unfinished))

        queue = deque()
        queued = set()
        index = start

        FINISHED_FLAG = np.iinfo(np.uint8).max
        print('start is ', index)
        while index is not None:
            # simulate completion at index
            unfinished[index] -= 1
            queue.append(index)
            queued.add(index)
            neighbors = self.get_all_neighbors(index, self.num_chunks)

            for neighbor in neighbors:
                new_value = unfinished[neighbor] - 1
                if new_value == 0:
                    new_value = FINISHED_FLAG
                unfinished[neighbor] = new_value


            # find next least work chunk to work on next
            focus_index = np.unravel_index(np.argmin(unfinished), unfinished.shape)
            # print('processed', index, 'now focusing on ', focus_index)
            # print(unfinished)

            new_neighbors = self.get_all_neighbors(focus_index, self.num_chunks)

            min_value = FINISHED_FLAG
            index = None
            for new_neighbor in new_neighbors:
                if new_neighbor not in queued:
                    value = unfinished[new_neighbor]
                    if value < min_value:
                        index = new_neighbor
                        min_value = value
        print(queue)

        return queue


    def get(self, start=None, dimensions=None):
        print('called get')
        if start is None:
            start = (0,) * len(self.num_chunks)

        queue = self.generate_queue(start)
        queue = deque(queue)

        path = np.zeros(self.num_chunks)


        print('The path to take is')
        for i in range(0, len(queue)):
            path[queue[i]] = i
        print(path)

        while len(queue) > 0:
            item = queue.popleft()
            yield item
            # yield queue.popleft()

        print('i am finished iterating')


class BlockProcessor:
    def __init__(self, block, on_next=None, on_error=None, on_completed=None, datasource_manager=None):
        self.block = block
        self.num_chunks = reduce(lambda x, y: x * y, block.num_chunks)
        self.completed = 0
        self.error = None
        self.on_next = on_next
        self.on_error = on_error
        self.on_completed = on_completed
        self.datasource_manager = datasource_manager

    @profile
    def process(self, processing_stream, start_slice=None, get_neighbors=None):
        print('Num_chunks %s' % (self.block.num_chunks,))
        if start_slice:
            start = self.block.slices_to_unit_index(start_slice)
        else:
            start = tuple([0] * len(self.block.bounds))

        min_step = pow(3, len(start))
        min_step = 9
        print('min_step is ', min_step)

        if get_neighbors is None:
            get_neighbors = self.block.get_all_neighbors


        # observable = Observable.from_(self.block.chunk_iterator(start)).controlled()
        # observable.request(27)

        completed = set()
        started_count = [0]
        counts = []
        previous_num_completed = [0]
        started = np.zeros(self.block.num_chunks, dtype=np.uint8)
        finished = np.zeros(self.block.num_chunks, dtype=np.uint8)

        def throttled_next(chunk):
            self._on_next(chunk)
            print('finally next', started_count)
            counts.append(started_count[0])
            print('counts are ', counts)

            finished[chunk.unit_index] = 1
            completed.add(chunk.unit_index)
            # for neighbor in get_neighbors(chunk):
            #     finished[neighbor.unit_index] = 1
            #     completed.add(neighbor.unit_index)

            completed_since = len(completed) - previous_num_completed[-1]
            print('completed since is', completed_since)
            if completed_since >= min_step or (completed_since + min_step > self.num_chunks):
                previous_num_completed[-1] = len(completed)
                # print('\n\n\nrequesting', min_step)
                # observable.request(min_step)

        iterator = self.block.chunk_iterator(start)
        current_process = psutil.Process()

        last_start_time = [time.time()]

        def get_memory_uss(process):
            # use uss to take account of shared library memory
            try:
                return process.memory_full_info().uss / 2. ** 30
            except psutil._exceptions.NoSuchProcess:
                return 0

        def get_memory_pss(process):
            # use uss to take account of shared library memory
            try:
                return process.memory_full_info().pss / 2. ** 30
            except psutil._exceptions.NoSuchProcess:
                return 0

        def get_buffer_info(name, chunk_buffer):
            if not chunk_buffer:
                return 0
            if  len(chunk_buffer.local_cache) == 0:
                keys = datas = []
            else:
                keys, chunks = zip(*chunk_buffer.local_cache.items())
                datas = [chunk.data for chunk in chunks if hasattr(chunk, 'data')]
            return get_mem_info(name, keys, datas)

        def get_mem_info(name, keys, datas):
            keys = list(keys)
            keys.sort()
            infos = [(x.shape, x.dtype, x.nbytes) for x in datas]
            memory_used = sum(info[2] for info in infos) / 2. ** 30
            if len(set(infos)) > 1:
                print('\n\n\nTHIS SHOULD NOT happen should not have more than one type of data... yet', infos)
            print('%s contains %s/%s (Futures ?: %s),entries of shape %s, total memory: %.3f GiB, entries:%s' % (
                name, len(infos), len(keys), len(keys) - len(infos),
                infos[0][1] if len(infos) else {}, memory_used, keys
            ))
            return memory_used

        def throttle_iterator(tick):
            processes = [current_process] + current_process.children(recursive=True)
            total_memory_pss = sum(map(get_memory_pss, processes))
            total_memory_uss = sum(map(get_memory_uss, processes))
            import objgraph
            print('\n growth:')
            objgraph.show_growth(limit=10)
            print('Total pss: %.3f' % total_memory_pss, 'Total uss: %.3f' % total_memory_uss, 'started:',
                  started_count[0], 'completed:', len(completed))
            memory_used = 0
            if self.datasource_manager is not None:
                input_buffer = self.datasource_manager.get_buffer(self.datasource_manager.input_datasource)
                output_buffer = self.datasource_manager.get_buffer(self.datasource_manager.output_datasource)
                output_final_buffer = self.datasource_manager.get_buffer(
                    self.datasource_manager.output_datasource_final)

                memory_used = 0
                memory_used += get_buffer_info('input_buffer', input_buffer)
                memory_used += get_buffer_info('output_buffer', output_buffer)
                memory_used += get_buffer_info('output_final_buffer', output_final_buffer)
                memory_used += get_mem_info('SparseOverlapRepository',
                                            self.datasource_manager.overlap_repository.datasources.keys(),
                                            self.datasource_manager.overlap_repository.datasources.values()
                                            )
                discrepancy = abs(total_memory_pss - memory_used)
                print('TOTAL expected memory used is %.3f but actual is pss %.3f, uss %.3f, discrepancy is: %.3f' % (
                    memory_used, total_memory_pss, total_memory_uss, discrepancy
                ))

            since_last_start = time.time() - last_start_time[0]
            overdue = since_last_start > 10
            if total_memory_pss > 6:
                print('RESETTIN PPE')
                self.datasource_manager.load_executor.shutdown(wait=True)
                self.datasource_manager.flush_executor.shutdown(wait=True)
                self.datasource_manager.load_executor = ProcessPoolExecutor()
                self.datasource_manager.flush_executor = ProcessPoolExecutor()
                print('DONE RESETTING PPE')
                gc.collect()
                processes = [current_process] + current_process.children(recursive=True)
                new_total_memory_pss = sum(map(get_memory_pss, processes))
                new_total_memory_uss = sum(map(get_memory_uss, processes))
                import objgraph
                print('\n growth:')
                objgraph.show_growth(limit=10)
                print('after reset ppe, Expected:', memory_used, 'Total pss: %.3f' % new_total_memory_pss, 'Total uss: %.3f'
                      % new_total_memory_uss, 'started:', started_count[0], 'completed:', len(completed),
                      'saved', new_total_memory_pss - total_memory_pss)

            if overdue:
                print('OVERDUE, shoving more chunks to see if it passes', overdue)
            if total_memory_pss < 10 or overdue:
                try:
                    chunk = next(iterator)
                except StopIteration:
                    return Observable.empty()
                started[chunk.unit_index] = 1
                started_count[0] += 1
                last_start_time[0] = time.time()
                print('Starting', chunk.unit_index, ' now:\n', started)
                return Observable.just(chunk)
            else:

                gc.collect()
                gc.collect()
                gc.collect()
                if self.datasource_manager is not None:
                    load_collect = [
                        self.datasource_manager.load_executor.submit(gc.collect)
                        for i in range(0, 12)
                    ]
                    flush_collect = [
                        self.datasource_manager.flush_executor.submit(gc.collect)
                        for i in range(0, 12)
                    ]

                    print('wait for flush to copmlete')
                    for future in as_completed(itertools.chain(load_collect, flush_collect)):
                        future.result()
                    print('done waiting for flush')
                    print('waiting to shutdown load')
                    self.datasource_manager.load_executor.shutdown(wait=True)
                    print('waiting to shutdown flush')
                    self.datasource_manager.flush_executor.shutdown(wait=True)
                gc.collect()
                gc.collect()

                processes = [current_process] + current_process.children(recursive=True)
                total_memory_pss = sum(map(get_memory_pss, processes))
                total_memory_uss = sum(map(get_memory_uss, processes))
                print('Tried to collect, memory now (pss):', total_memory_pss, 'memory now (uss):', total_memory_uss)
                raise Exception('fu')
                return Observable.never()

        print('about to begin')
        (
            # observable
            Observable.interval(100)
            .flat_map(throttle_iterator)
            # .filter(lambda x: x is not None)
            .flat_map(processing_stream)
            .to_blocking()
            .blocking_subscribe(throttled_next, on_error=self._on_error, on_completed=self._on_completed)
            # .subscribe(throttled_next, on_error=self._on_error, on_completed=self._on_completed)
        )
        # import time
        # sleepy = 20
        # print('sleeping for ', sleepy)
        # time.sleep(sleepy)
        # print('done sleeping for', sleepy)
        assert False

    def _on_completed(self):
        print('Finished processing', self.num_chunks)
        if self.on_completed is not None:
            self.on_completed()


    def _on_error(self, error):
        print('\n\n\n\n************************************error************************************\n\n')
        self.error = error
        traceback.print_exception(None, error, error.__traceback__)
        if self.on_error is not None:
            self.on_error(error)
        raise error

    def _on_next(self, chunk, data=None):
        self.completed += 1
        print('****** %s--%s %s. %s of %s done ' % (datetime.now(), current_thread().name, chunk.unit_index,
                                                    self.completed, self.num_chunks))
        if self.on_next is not None:
            self.on_next(chunk)
