import traceback
from collections import deque
from datetime import datetime
from functools import reduce
from threading import current_thread
from chunkblocks.iterators import UnitIterator
from chunkflow.streams import blocking_subscribe
import itertools
import functools
import psutil
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
    def __init__(self, block, on_next=None, on_error=None, on_completed=None):
        self.block = block
        self.num_chunks = reduce(lambda x, y: x * y, block.num_chunks)
        self.completed = 0
        self.error = None
        self.on_next = on_next
        self.on_error = on_error
        self.on_completed = on_completed

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
        finished = np.zeros(self.block.num_chunks, dtype=np.uint8)

        def throttled_next(chunk):
            self._on_next(chunk)
            print('finally next', started_count)
            counts.append(started_count[0])
            print('counts are ', counts)

            finished[chunk.unit_index] = 1
            completed.add(chunk.unit_index)
            for neighbor in get_neighbors(chunk):
                finished[neighbor.unit_index] = 1
                completed.add(neighbor.unit_index)

            completed_since = len(completed) - previous_num_completed[-1]
            print('complted_since is', completed_since)
            if completed_since >= min_step or (completed_since + min_step > self.num_chunks):
                previous_num_completed[-1] = len(completed)
                # print('\n\n\nrequesting', min_step)
                # observable.request(min_step)

        iterator = self.block.chunk_iterator(start)
        current_process = psutil.Process()

        def get_memory(process):
            return process.memory_info()[0] / 2. ** 30

        def throttle_iterator(x):
            memory = get_memory(current_process)
            children_memory = sum(map(get_memory, current_process.children(recursive=True)))
            total_memory = memory + children_memory

            print('Memory used:', memory, 'Children:', children_memory, 'Total:', total_memory)
            if total_memory < 2:
                return next(iterator)
            else:
                gc.collect()

        print('about to begin')
        (
            # observable
            Observable.interval(100)
            .map(throttle_iterator)
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
