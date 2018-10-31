import traceback
from collections import deque
from datetime import datetime
from functools import reduce
from threading import current_thread
from chunkblocks.iterators import UnitIterator
from chunkflow.streams import blocking_subscribe
import itertools
import functools
import linecache
import tracemalloc
from memory_profiler import profile
from concurrent.futures import as_completed, ProcessPoolExecutor
import psutil
import time
import os
import gc
import numpy as np
import memorytools
from rx import Observable

SENTINEL = 1337



def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

class ReadyNeighborIterator(UnitIterator):
    def __init__(self, num_chunks, block=None, datasource_block=None):
        self.num_chunks = num_chunks
        self.block = block
        self.datasource_block = datasource_block

    # def get_all_neighbors(self, index, max=None):
    #     neighbors = set(super().get_all_neighbors(index, max))

    #     for neighbor in list(neighbors):
    #         slices = self.block.unit_index_to_slices(neighbor)
    #         datasource_chunks = self.datasource_block.slices_to_chunks(slices)
    #         for datasource_chunk in datasource_chunks:
    #             neighbors.update(set(self.block.slices_to_unit_indices(datasource_chunk.slices)))
    #     # slices = self.block.unit_index_to_slices(index)
    #     # datasource_chunks = self.datasource_block.slices_to_chunks(slices)
    #     # for datasource_chunk in datasource_chunks:
    #     #     neighbors.update(set(self.block.slices_to_unit_indices(datasource_chunk.slices)))
    #     return sorted(neighbors)


    def generate_queue(self, start):
        print('generating queue')
        queue_1 = deque()
        queued_1 = set()
        queue_2 = deque()
        queued_2 = set()

        unfinished = np.zeros(self.num_chunks, dtype=np.uint16)
        it = np.nditer(unfinished, flags=['multi_index'])
        while not it.finished:
            index = it.multi_index
            relevant_neighbors = set()
            for neighbor in super().get_all_neighbors(index, self.num_chunks):
                for neighbor_neighbor in super().get_all_neighbors(neighbor, self.num_chunks):
                    if neighbor_neighbor not in relevant_neighbors:
                        relevant_neighbors.add(neighbor_neighbor)

            chunk_slices = self.block.unit_index_to_slices(index)
            for datasource_chunk in self.datasource_block.slices_to_chunks(chunk_slices):
                for chunk in self.block.slices_to_chunks(datasource_chunk.slices):
                    if chunk.unit_index not in relevant_neighbors:
                        relevant_neighbors.add(chunk.unit_index)

            for relevant_neighbor in relevant_neighbors:
                unfinished[relevant_neighbor] += 1

            # unfinished[it.multi_index] += len(self.get_all_neighbors(it.multi_index, self.num_chunks))
            it.iternext()

        print('unfinished is\n', unfinished)


        chunk_slices = self.block.unit_index_to_slices(start)
        print('begin with ', start, queue_1, len(queue_1))
        queue_1.append(start)
        queued_1.add(start)

        FINISHED_FLAG = np.iinfo(np.uint16).max

        def euclidean_distance(left, right):
            return sum(map(lambda lr: abs(lr[0] - lr[1]), zip(left, right)))

        things = []
        things_2 = [0]
        while len(queue_1) > 0:
            index = queue_1.popleft()
            queue_2.append(index)
            queued_2.add(index)
            chunk_slices = self.block.unit_index_to_slices(index)

            relevant_neighbors = dict()

            # add neighbor neighbors as upload dependency
            for neighbor in super().get_all_neighbors(index, self.num_chunks):
                for neighbor_neighbor in super().get_all_neighbors(neighbor, self.num_chunks):
                    relevant_neighbors[neighbor_neighbor] = unfinished[neighbor_neighbor]

            # add flush dependencies
            for datasource_chunk in self.datasource_block.slices_to_chunks(chunk_slices):
                for chunk in self.block.slices_to_chunks(datasource_chunk.slices):
                    relevant_neighbors[chunk.unit_index] = unfinished[chunk.unit_index]

            for relevant_neighbor in relevant_neighbors.keys():
                new_value = unfinished[relevant_neighbor] - 1

                if new_value == 0:
                    things.append(len(queue_2) - things_2[-1])
                    things_2.append(len(queue_2))
                    new_value = FINISHED_FLAG
                unfinished[relevant_neighbor] = new_value

            # while len(queue_1) > 0:
            #     q1 = queue_1.pop()
            #     relevant_neighbors[q1] = unfinished[q1]

            for relevant_neighbor, value in sorted(relevant_neighbors.items(),
                                                   key=lambda item: (euclidean_distance(item[0], index))):
                if relevant_neighbor not in queued_1 and relevant_neighbor not in queued_2:
                # if relevant_neighbor not in queued_2:
                    queue_1.append(relevant_neighbor)
                    queued_1.add(relevant_neighbor)

        print('unfinished\n', unfinished)
        print('queued1 is ', queue_1)
        print('queued2 is ', queue_2)

        directions = np.zeros(self.num_chunks)
        print(len(queue_2))
        print(np.prod(self.num_chunks))
        for i in range(0, len(queue_2)):
            directions[queue_2[i]] = i

        print('\n\n\ndirections are')
        print(directions)
        print('things are', things)
        print('things2 are', things_2)
        return queue_2


    def generate_queuea(self, start):
        """
        Ghetto space filling curve that's kinda like z-order, but probably worse and not performant
        """
        print('generating queue')
        unfinished = np.ones(self.num_chunks, dtype=np.uint8)
        it = np.nditer(unfinished, flags=['multi_index'])
        while not it.finished:
            unfinished[it.multi_index] += len(self.get_all_neighbors(it.multi_index, self.num_chunks))
            it.iternext()

        print(unfinished, np.any(unfinished))

        queue = deque()
        queued = set()
        index = start

        FINISHED_FLAG = np.iinfo(np.uint8).max
        print('start is ', index, 'neighbors are', sorted(self.get_all_neighbors(index, self.num_chunks)))
        print(len(self.get_all_neighbors(index, self.num_chunks)))
        last_finished_counter = 0
        last_finished = []


        while index is not None:
            # simulate completion at index
            unfinished[index] -= 1
            queue.append(index)
            queued.add(index)
            neighbors = self.get_all_neighbors(index, self.num_chunks)

            last_finished_counter += 1
            for neighbor in neighbors:
                new_value = unfinished[neighbor] - 1
                if new_value == 0:
                    new_value = FINISHED_FLAG
                    last_finished.append(last_finished_counter)
                    last_finished_counter = 0
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

        # delete me
        directions = np.zeros(self.num_chunks)
        for i in range(0, len(queue)):
            directions[queue[i]] = i

        print('\n\n\ndirection sare')
        print(directions)
        print(queue)
        print('last finished:', last_finished)
        print(max(last_finished))
        # delete me

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




from memory_profiler import profile

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
        previous_num_completed = [0]
        started = np.zeros(self.block.num_chunks, dtype=np.uint8)
        finished = np.zeros(self.block.num_chunks, dtype=np.uint8)
        start_time = time.time()

        def throttled_next(chunk):
            memorytools.summarize_objects()
            self._on_next(chunk)
            self.datasource_manager.print_cache_stats()
            print('completing ', started_count)

            finished[chunk.unit_index] = 1
            completed.add(chunk.unit_index)

            # delete me
            completed_since = len(completed) - previous_num_completed[0]
            print('completed since is', completed_since)
            if completed_since >= min_step or (completed_since + min_step > self.num_chunks):
                previous_num_completed[0] = len(completed)
            # delete me

        iterator = self.block.chunk_iterator(start)
        current_process = psutil.Process()

        last_start_time = [time.time()]
        force_push = [0]

        def throttle_iterator(tick):
            print('started', started_count[0], 'completed:', len(completed), 'Elapsed', time.time() - start_time)
            completed_since = len(completed) - previous_num_completed[0]
            since_last_start = time.time() - last_start_time[0]
            overdue = since_last_start > 10

            if overdue and force_push[0] == 0:
                print('forcing through additional 10')
                force_push[0] = 10

            if started_count[0] > 100 and completed_since < 1 and force_push[0] == 0:
                print('throttling \n\n\n')
                return Observable.empty().filter(lambda x: x is not None)


            # if discrepancy > 2.5:
            #     print('RESETTIN PPE')
            #     if self.datasource_manager.load_executor is not None:
            #         self.datasource_manager.load_executor.shutdown(wait=True)
            #         self.datasource_manager.load_executor = ProcessPoolExecutor()
            #     if self.datasource_manager.flush_executor is not None:
            #         self.datasource_manager.flush_executor.shutdown(wait=True)
            #         self.datasource_manager.flush_executor = ProcessPoolExecutor()
            #     print('DONE RESETTING PPE')
            #     gc.collect()
            #     new_total_memory_pss, new_total_memory_uss = print_memory(current_process)
            #     import objgraph
            #     print('\n growth:')
            #     objgraph.show_growth(limit=10)
            #     print('after reset ppe, Expected:', memory_used, 'Total pss: %.3f' % new_total_memory_pss, 'Total uss: %.3f'
            #           % new_total_memory_uss, 'saved', new_total_memory_pss - total_memory_pss)

            if True or total_memory_pss < 10:
                try:
                    chunk = next(iterator)
                except StopIteration:
                    return Observable.just(SENTINEL)
                # started[chunk.unit_index] = 1
                started_count[0] += 1
                last_start_time[0] = time.time()
                if force_push[0]:
                    force_push[0] -= 1
                print('Starting', chunk.unit_index, ' now:\n')#, started)
                return Observable.just(chunk)
            else:
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

                return Observable.empty()

        print('about to begin')
        started = [0]
        def inc():
            started[0] += 1

        def dec():
            started[0] -= 1

        tracemalloc.start()
        # from chunkblocks.global_offset_array import GlobalOffsetArray
        # __import__('ipdb').set_trace()

        (
            # observable
            Observable.interval(50)
            .flat_map(throttle_iterator)
            # Observable.from_(self.block.chunk_iterator(start))
            .do_action(lambda _: inc() or print('just started, need to process:', started[0], 'elapsed', time.time() -
                                                start_time))
            .do_action(lambda x: self.datasource_manager.print_cache_stats())
            .take_while(lambda x: x is not SENTINEL)
            .flat_map(processing_stream)
            .do_action(lambda _: dec() or print('just finished, still need to process:', started[0], 'elapsed',
                                                time.time() - start_time))
            .to_blocking()
            .blocking_subscribe(throttled_next, on_error=self._on_error, on_completed=self._on_completed)
            # .subscribe(throttled_next, on_error=self._on_error, on_completed=self._on_completed)
        )

        # import time
        # sleepy = 20
        # print('sleeping for ', sleepy)
        # time.sleep(sleepy)
        # print('done sleeping for', sleepy)

    def _on_completed(self):
        gc.collect()
        print('Finished processing', self.num_chunks)
        memorytools.summarize_objects()

        if self.on_completed is not None:
            self.on_completed()
        self.datasource_manager.print_cache_stats()
        snap = tracemalloc.take_snapshot()
        display_top(snap)
        # __import__('pdb').set_trace()


    def _on_error(self, error):
        print('\n\n\n\n************************************error************************************\n\n')
        self.error = error
        traceback.print_exception(None, error, error.__traceback__)
        if self.on_error is not None:
            self.on_error(error)
        raise error

    def _on_next(self, chunk, data=None):
        self.completed += 1
        # print('RESETTIN PPE')
        # if self.datasource_manager.load_executor is not None:
        #     self.datasource_manager.load_executor.shutdown(wait=True)
        #     self.datasource_manager.load_executor = ProcessPoolExecutor()
        # if self.datasource_manager.flush_executor is not None:
        #     self.datasource_manager.flush_executor.shutdown(wait=True)
        #     self.datasource_manager.flush_executor = ProcessPoolExecutor()
        # print('DONE RESETTING PPE')
        print('****** %s--%s %s. %s of %s done ' % (datetime.now(), current_thread().name, chunk.unit_index,
                                                    self.completed, self.num_chunks))
        if self.on_next is not None:
            self.on_next(chunk)
