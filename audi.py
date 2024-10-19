import functools
from contextlib import contextmanager


class MyFuncTracker(object):
    """A decorator class to track function inputs and outputs.

    Store recorded calls in attribute `call_tape`, a list of tuples
    representing (inputs_k, outputs_k, func_k).

    Args:
        do_track (bool): A boolean flag to determine whether tracking is enabled.
    """

    def __init__(self, do_track: bool):
        self.do_track = do_track
        self.reset()

    def reset(self):
        """Reset the call tape to empty."""
        self.call_tape = []

    def __call__(self, func):
        """Wrap the function to track inputs and outputs in `self.call_tape`."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.do_track:
                inputs = args
                output = func(*args, **kwargs)
                self.call_tape.append((inputs, output, func))
                return output
            else:
                return func(*args, **kwargs)

        return wrapper

    @contextmanager
    def track_func(self, do_track: bool):
        """Context manager to enable or disable tracking within a block."""
        try:
            self.do_track = do_track
            yield
        finally:
            self.do_track = False


# initialize a global function tracker
my_func_tracker = MyFuncTracker(do_track=True)


@my_func_tracker
def Add(a, b):
    return a + b


@my_func_tracker
def Multiply(a, b):
    return a * b


class MyTensor(object):
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        new_value = Add(self.value, other.value)
        return MyTensor(new_value)

    def __mul__(self, other):
        new_value = Multiply(self.value, other.value)
        return MyTensor(new_value)


x = MyTensor(1.0)
y = MyTensor(2.0)

my_func_tracker.reset()
with my_func_tracker.track_func(True):
    # do computations and track
    z = x + y
    z = x * z

# extract computation history
for call_inputs, call_output, func in my_func_tracker.call_tape:
    print(f"Function: {func.__name__}, Inputs: {call_inputs}, Output: {call_output}")
