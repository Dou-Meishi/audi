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


class MyTensor(object):
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return multiply(self, other)

    def __repr__(self):
        return repr(self.value)


@my_func_tracker
def _add(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value + b.value)


@my_func_tracker
def _multiply(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value * b.value)


class MyFunction(object):
    def __init__(self, func, func_vjp=None, func_jvp=None):
        self.func = func
        self.vjp = func_vjp
        self.jvp = func_jvp

    def __call__(self, *args, **kws):
        return self.func(*args, **kws)

    def name(self) -> str:
        return self.func.__name__


add = MyFunction(_add)
multiply = MyFunction(_multiply)


def main():
    x = MyTensor(1.0)
    y = MyTensor(2.0)

    my_func_tracker.reset()
    with my_func_tracker.track_func(True):
        # do computations and track
        z = x + y
        z = x * z

    # extract computation history
    for call_inputs, call_output, func in my_func_tracker.call_tape:
        print(
            f"Function: {func.__name__}, Inputs: {call_inputs}, Output: {call_output}"
        )


if __name__ == "__main__":
    main()
