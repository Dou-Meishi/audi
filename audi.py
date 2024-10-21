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
        return mul(self, other)

    def __rmul__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return mul(self, other)

    def __sub__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return sub(self, other)

    def __rsub__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return sub(other, self)

    def __truediv__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return div(self, other)

    def __rtruediv__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return div(other, self)

    def __neg__(self):
        return neg(self)

    def __repr__(self):
        return repr(self.value)

    def dot(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return dot(self, other)

    def sin(self):
        return sin(self)

    def cos(self):
        return cos(self)

    def exp(self):
        return exp(self)

    def log(self):
        return log(self)


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


def _mul(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value * b.value)


def _mul_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    a, b = inputs
    return (b * grad_outputs, a * grad_outputs)


def _mul_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    a, b = inputs
    return (b * grad_outputs, a * grad_outputs)


def _dot(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value.dot(b.value))


def _dot_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    a, b = inputs
    return b * grad_outputs, a * grad_outputs


def _dot_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    a, b = inputs
    return b.dot(grad_outputs), a.dot(grad_outputs)


def _sin(a: MyTensor) -> MyTensor:
    return MyTensor(np.sin(a.value))


def _sin_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs * inputs[0].cos(),)


def _sin_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs * inputs[0].cos(),)


def _cos(a: MyTensor) -> MyTensor:
    return MyTensor(np.cos(a.value))


def _cos_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (-grad_outputs * inputs[0].sin(),)


def _cos_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (-grad_outputs * inputs[0].sin(),)


def _exp(a: MyTensor) -> MyTensor:
    return MyTensor(np.exp(a.value))


def _exp_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs * outputs,)


def _exp_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs * outputs,)


def _sub(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value - b.value)


def _sub_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return [grad_outputs, -grad_outputs]


def _sub_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return [grad_outputs, -grad_outputs]


def _neg(a: MyTensor) -> MyTensor:
    return MyTensor(-a.value)


def _neg_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (-grad_outputs,)


def _neg_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (-grad_outputs,)


def _log(a: MyTensor) -> MyTensor:
    return MyTensor(np.log(a.value))


def _log_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs / inputs[0],)


def _log_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs / inputs[0],)


def _div(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value / b.value)


def _div_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    _, b = inputs
    return grad_outputs / b, -grad_outputs * outputs / b


def _div_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    _, b = inputs
    return grad_outputs / b, -grad_outputs * outputs / b


add = MyFunction("Add", _add, func_vjp=_add_vjp, func_jvp=_add_jvp)
mul = MyFunction("Mul", _mul, func_vjp=_mul_vjp, func_jvp=_mul_jvp)
dot = MyFunction("Dot", _dot, func_vjp=_dot_vjp, func_jvp=_dot_jvp)
sin = MyFunction("Sin", _sin, func_vjp=_sin_vjp, func_jvp=_sin_jvp)
cos = MyFunction("Cos", _cos, func_vjp=_cos_vjp, func_jvp=_cos_jvp)
exp = MyFunction("Exp", _exp, func_vjp=_exp_vjp, func_jvp=_exp_jvp)
sub = MyFunction("Sub", _sub, func_vjp=_sub_vjp, func_jvp=_sub_jvp)
neg = MyFunction("Neg", _neg, func_vjp=_neg_vjp, func_jvp=_neg_jvp)
log = MyFunction("Log", _log, func_vjp=_log_vjp, func_jvp=_log_jvp)
div = MyFunction("Div", _div, func_vjp=_div_vjp, func_jvp=_div_jvp)


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

    a = MyTensor(np.random.randn(3))
    b = MyTensor(np.random.randn(3))
    v = MyTensor(np.random.randn(3))

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
