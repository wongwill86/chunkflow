import traceback
from datetime import datetime
from functools import reduce
from threading import current_thread
from chunkblocks.iterators import UnitIterator
from chunkflow.streams import blocking_subscribe

from rx import Observable


class ReadyNeighborIterator(UnitIterator):
    def __init__(self, block):
        self.block = block
        self.num_chunks = block.num_chunks

    def get(self, start=None, dimensions=None):
        if start is None:
            start = (0,) * len(self.num_chunks)
        queue = []
        queue.append(start)

        queued = set()
        completed = set()
        def mark_done(marked_index):
            completed.add(marked_index)
            candidate_list = []
            for neighbor_index in self.get_all_neighbors(marked_index, self.num_chunks):
                candidates = []
                # schedule neighbors of neighbors if necessary
                for neighbor_neighbor_index in self.get_all_neighbors(neighbor_index, self.num_chunks):
                    if neighbor_neighbor_index not in completed:
                        candidates.append(neighbor_neighbor_index)
                candidate_list.append(candidates)

            candidate_list.sort(key=len)
            to_queue = [
                candidate for candidates in candidate_list for candidate in candidates if candidate not in queued
            ]

            queue.extend(to_queue)
            for item in to_queue:
                queued.add(item)

        mark_done(start)

        mark = None
        while len(queue) > 0:
            if mark is None:
                mark = yield self.block.unit_index_to_chunk(queue.pop())
            else:
                mark_done(mark.unit_index)
                mark = yield None


class BlockProcessor:
    def __init__(self, block, iterator=None, on_next=None, on_error=None, on_completed=None):
        self.block = block
        self.num_chunks = reduce(lambda x, y: x * y, block.num_chunks)
        self.completed = 0
        self.error = None
        self.on_next = on_next
        self.on_error = on_error
        self.on_completed = on_completed
        self.iterator = iterator

    def process(self, processing_stream, start_slice=None):
        print('Num_chunks %s' % (self.block.num_chunks,))
        if start_slice:
            start = self.block.slices_to_unit_index(start_slice)
        else:
            start = tuple([0] * len(self.block.bounds))

        if self.iterator is None:
            iterator = self.block.chunk_iterator(start)

        (
            Observable.from_(iterator)
            .do_action(lambda ugh: print('ugh is ', ugh))
            .flat_map(processing_stream)
            .to_blocking()
            .blocking_subscribe(self._on_next, on_error=self._on_error, on_completed=self._on_completed)
        )

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
