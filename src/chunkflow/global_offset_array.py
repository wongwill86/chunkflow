import numbers

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

OUT = 'out'


class GlobalOffsetArray(np.ndarray, NDArrayOperatorsMixin):
    """
    A simple VIEW CAST of a given ndarray that is addressed via global coordinates. Negative wraparound indices are NOT
    supported (i.e. used for printing out)

    See below link for explanations of __new__ and __array_finalize__!
    https://docs.scipy.org/doc/numpy-dev/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    """# noqa
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __new__(cls, input_array, global_offset=None):
        obj = np.asarray(input_array).view(cls)
        if global_offset is None:
            global_offset = tuple([0] * input_array.ndim)

        obj.global_offset = tuple(global_offset)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.global_offset = getattr(obj, 'global_offset', None)

    def _to_internal_slices(self, index):
        """
        Convert given index into the index used in the internal ndarray. Does NOT support end slicing and wrap around
        negative indices. Throw an error if computed internal indices are outside the range of the data.
        """
        shape = self.shape
        internal_index = ()
        new_global_offset = ()

        for dimension, item in enumerate(index):
            offset = self.global_offset[dimension]
            length = self.shape[dimension]

            if isinstance(item, slice):
                new_item = slice(item.start - offset, item.stop - offset, item.step)
                new_global_offset += (new_item.start + offset,)

                if (new_item.start < 0 or new_item.start >= length or new_item.stop < 1 or new_item.stop > length):
                    raise IndexError('Accessing slice [%s, %s) at dimension %s out of data bounds [%s, %s) '
                                     'shape: %s global_offset: %s ' % (
                                         new_item.start + offset, new_item.stop + offset, dimension,
                                         offset, offset + length, shape, self.global_offset))
            else:
                new_item = item - offset

                if new_item < 0 or new_item > length:
                    raise IndexError('Accessing index %s at dimension %s out of data bounds [%s , %s) '
                                     'shape: %s global_offset: %s ' % (
                                         new_item, dimension, offset, offset + length, shape, self.global_offset))

            internal_index += (new_item,)
        return (internal_index, new_global_offset)

    def __getitem__(self, index):
        """
        Access the array based on global coordinates. If we receive a tuple, it means we are slicing.
        When we slice, calculate the actual coordinates stored
        """
        if isinstance(index, tuple):
            internal_index, new_global_offset = self._to_internal_slices(index)
        else:
            internal_index = index
            new_global_offset = None

        new_from_template = super(GlobalOffsetArray, self).__getitem__(internal_index)
        if hasattr(new_from_template, 'global_offset'):
            new_from_template.global_offset = new_global_offset
        return new_from_template

    def __setitem__(self, index, value):
        """
        Access the array based on global coordinates. If we receive a tuple, it means we are slicing.
        When we slice, calculate the actual coordinates stored
        """
        # use view instead of super because super will call the overriden __getitem__ function

        # self[index].view(np.ndarray)[:] = value
        if isinstance(index, tuple):
            internal_index, _ = self._to_internal_slices(index)
        else:
            internal_index = index

        self.view(np.ndarray).__setitem__(internal_index, value)

    def __str__(self):
        """
        Overwrite string conversion to create a view instead of calling super. Super will call with the overridden
        __getitem__ function which will not work
        """
        return '%s, global_offset: %s' % (self.view(np.ndarray).__str__(), self.global_offset)

    def __repr__(self):
        """
        Overwrite string conversion to create a view instead of calling super. Super will call with the overridden
        __getitem__ function which will not work
        """
        return self.view(np.ndarray).__repr__()

    def bounds(self):
        """
        Get slices that are bounds of the available data
        """
        return tuple(slice(offset, offset + self.shape[dimension]) for dimension, offset in
                     enumerate(self.global_offset))

    def is_contained_within(self, other):
        """
        Check to see if this volume is contained within other
        """
        if self.ndim != other.ndim:
            raise ValueError("Checking with incompatible dimensions. self: %s, other: %s" % (self.shape, other.shape))

        self_bounds = self.bounds()
        other_bounds = other.bounds()
        return all(other_slice.start <= self_slice.start and self_slice.start <= other_slice.stop and
                   other_slice.start <= self_slice.stop and self_slice.stop <= other_slice.stop
                   for self_slice, other_slice in zip(self_bounds, other_bounds))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): # noqa: C901
        """
        Enable injection of customized indexing for ufunc operations
        Must defer to the implementation of the ufunc on unwrapped values to avoid infinite loop
        https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
        """
        in_place = OUT in kwargs
        for x in inputs + kwargs.get(OUT, ()):
            # Use GlobalOffsetArray instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle GlobalOffsetArray objects.
            if not isinstance(x, self._HANDLED_TYPES + (GlobalOffsetArray,)):
                return NotImplemented

        # global offset to use in the result
        global_offset = None
        if len(inputs) == 2:
            left = inputs[0]
            right = inputs[1]

            left_is_global_offset_array = isinstance(left, GlobalOffsetArray)
            right_is_global_offset_array = isinstance(right, GlobalOffsetArray)
            if left_is_global_offset_array and right_is_global_offset_array:
                smaller = larger = None
                left_in_right = left.is_contained_within(right)
                right_in_left = right.is_contained_within(left)

                if not left_in_right and not right_in_left:
                    raise ValueError("Incorrect overlapping indices. Left bounds: %s, Right bounds: %s" % (
                        left.bounds(), right.bounds()))

                if left_in_right and right_in_left:
                    # same bounds/size
                    global_offset = left.global_offset
                else:
                    smaller = left if left_in_right else right
                    larger = right if left_in_right else left

                    sub_left = left[smaller.bounds()]
                    sub_right = right[smaller.bounds()]

                    sub_inputs = (sub_left, sub_right)
                    sub_kwargs = {}

                    if in_place:
                        if left is smaller:
                            raise ValueError("In-place operation must have Left (shape: %s) larger or equal than Right"
                                             " (shape: %s)" % (left.shape, right.shape))
                        sub_kwargs[OUT] = tuple(o[smaller.bounds()] for o in kwargs[OUT])
                        getattr(ufunc, method)(*sub_inputs, **sub_kwargs)
                        return kwargs[OUT]
                    else:
                        # Return a copy of the larger operand and perform in place on the sub_array of that copy
                        sample_type = type(getattr(ufunc, method)(sub_left.item(0), sub_right.item(1)))
                        result = larger.astype(sample_type)
                        sub_kwargs[OUT] = (result[smaller.bounds()])
                        sub_result = getattr(ufunc, method)(*sub_inputs, **sub_kwargs)
                        result[smaller.bounds()] = sub_result

                        return result

            elif left_is_global_offset_array:
                global_offset = left.global_offset
            elif right_is_global_offset_array:
                global_offset = right.global_offset

            inputs = (left, right)

        # Must defer to the implementation of the ufunc on unwrapped values to avoid infinite loop
        inputs = tuple(i.view(np.ndarray) if isinstance(i, GlobalOffsetArray) else i for i in inputs)
        if in_place:
            kwargs[OUT] = tuple(o.view(np.ndarray) if isinstance(o, GlobalOffsetArray) else o for o in kwargs[OUT])

        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x, global_offset=global_offset) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result, global_offset=global_offset)
