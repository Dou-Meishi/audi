from collections.abc import Callable
import functools
from contextlib import contextmanager

import numpy as np


class MyFunction(object):
    """Functions with vjp and jvp as attributes."""

    def __init__(self, name, func, func_vjp=None, func_jvp=None):
        self.name = name
        self.func = func
        self.vjp = func_vjp
        self.jvp = func_jvp

    def __call__(self, *args, **kws):
        return self.func(*args, **kws)


class MyFuncTracker(object):
    """A decorator class to track function inputs and outputs.
    Designed for MyFunction class.

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
        """Wrap the function to track inputs and outputs in `self.call_tape`.
        Expect func receive self as its first argument."""

        @functools.wraps(func)
        def wrapper(objself, *args, **kwargs):
            if self.do_track:
                inputs = args
                output = func(objself, *args, **kwargs)
                self.call_tape.append((inputs, output, objself))
                return output
            else:
                return func(objself, *args, **kwargs)

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
# apply it to MyFunction class
MyFunction.__call__ = my_func_tracker(MyFunction.__call__)


class MyTensor(object):
    def __init__(self, value, grad=0):
        self.value = np.asarray(value)
        self.grad = grad

    def __add__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return add(self, other)

    def __radd__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return add(self, other)

    def __mul__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return multiply(self, other)

    def __rmul__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return multiply(self, other)

    def __repr__(self):
        return repr(self.value)


def _add(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value + b.value)


def _add_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs for _ in inputs)


def _add_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs for _ in inputs)


def _multiply(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value * b.value)


def _multiply_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    a, b = inputs
    return (b * grad_outputs, a * grad_outputs)


def _multiply_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    a, b = inputs
    return (b * grad_outputs, a * grad_outputs)


add = MyFunction("Add", _add, func_vjp=_add_vjp, func_jvp=_add_jvp)
multiply = MyFunction("Multiply", _multiply, func_vjp=_multiply_vjp, func_jvp=_multiply_jvp)


def reverseAD(
    f: Callable[[list[MyTensor]], MyTensor],
    inputs: list[MyTensor],
    v: MyTensor,
) -> list[MyTensor]:
    """Use reverse-mode AD to compute the vector-Jacobian product of f.
    Return the gradient of dot(f, v) evaluated at inputs.

    Args
    ----
    - `f`: The function to be differentiated.

    - `inputs`: Inputs of `f`.

    - `v`: Any tensor matches the dim of `f`. Default to all one tensor.
           In the default case, this function effectively differentiates
           the sum of f's components.

    Note
    ----
    Gradients are accumulated in tensor's `grad` attribute, which is
    zero by default. However, this function does not check whether
    `grad` is zero or not. It simply accumulates all gradient in it."""
    my_func_tracker.reset()
    with my_func_tracker.track_func(True):
        # do computations and track
        y = f(*inputs)

    # extract computation history
    tape = my_func_tracker.call_tape
    # backpropagate gradient starting at y
    reverseAD_along_tape(y, tape, v)
    return [x.grad for x in inputs]


def reverseAD_along_tape(y, call_tape, v):
    """Backpropagate gradient starting at y. Initially y.grad is set to v."""
    y.grad = v
    for k_inputs, k_outputs, k_phi in reversed(call_tape):
        # chain rule
        grad_inputs = k_phi.vjp(k_inputs, k_outputs, k_outputs.grad)
        # accumulate grad
        for x, grad in zip(k_inputs, grad_inputs):
            x.grad += grad


def main():
    print("Test with function f(a, b) = a*(a+b)")

    def test_f1(a, b):
        z = a + b
        z = a * z
        return z

    def test_f1_vjp(a, b, v):
        grad_a = v * (2 * a + b)
        grad_b = v * a
        return grad_a, grad_b

    a = MyTensor(np.random.randn())
    b = MyTensor(np.random.randn())
    v = MyTensor(np.random.randn())

    grad_a, grad_b = reverseAD(test_f1, [a, b], v)
    expected_grad_a, expected_grad_b = test_f1_vjp(a, b, v)
    match_a = np.allclose(grad_a.value, expected_grad_a.value)
    match_b = np.allclose(grad_b.value, expected_grad_b.value)

    print(f"Gradient of a: {grad_a}. Matches expected value: {match_a}")
    print(f"Gradient of b: {grad_b}. Matches expected value: {match_b}")

    # examine computation history
    # for call_inputs, call_output, func in my_func_tracker.call_tape:
    #     print(f"Function: {func.name}, Inputs: {call_inputs}, Output: {call_output}")


if __name__ == "__main__":
    main()
