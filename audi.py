from collections.abc import Callable
import functools
from contextlib import contextmanager
from typing import Union

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

        self.debug = False

    def reset(self):
        """Reset the call tape to empty."""
        self.call_tape = []

    def __call__(self, func):
        """Wrap the function to track inputs and outputs in `self.call_tape`.
        Expect func receive self as its first argument."""

        @functools.wraps(func)
        def wrapper(objself, *args, **kwargs):
            if self.do_track:
                output = func(objself, *args, **kwargs)
                self.call_tape.append((args, output, objself, kwargs))

                if self.debug:
                    print(f"Function: {objself.name} (with kwargs {kwargs})")
                    print(f"\tInputs: {args}")
                    print(f"\tOutput: {output}")

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
        a, b = MyTensor.broadcast(self, other)
        return add(a, b)

    def __radd__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        a, b = MyTensor.broadcast(self, other)
        return add(b, a)

    def __mul__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        a, b = MyTensor.broadcast(self, other)
        return mul(a, b)

    def __rmul__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        a, b = MyTensor.broadcast(self, other)
        return mul(b, a)

    def __matmul__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        # as matmul can only deal with matrix-matrix multiplication,
        # we have to convert vectors to matrix manually
        # and then convert the result back
        if self.ndim == 2 and other.ndim == 2:
            # matrix-matrix multiplication
            return matmul(self, other)
        elif self.ndim == 2 and other.ndim == 1:
            # matrix-vector multiplication
            other = other.as_row_vector().T
            return matmul(self, other).sum(dim=1)
        elif self.ndim == 1 and other.ndim == 2:
            # vector-matrix multiplication
            self = self.as_row_vector()
            return matmul(self, other).sum(dim=0)
        elif self.ndim == 1 and other.ndim == 1:
            # vector dot
            self = self.as_row_vector()
            other = other.as_row_vector().T
            return matmul(self, other).sum()
        else:
            raise ValueError(f"Invalid shape {self.shape} @ {other.shape} for Matmul")

    def __rmatmul__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        return other @ self

    def __sub__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        a, b = MyTensor.broadcast(self, other)
        return sub(a, b)

    def __rsub__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        a, b = MyTensor.broadcast(self, other)
        return sub(b, a)

    def __truediv__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        a, b = MyTensor.broadcast(self, other)
        return div(a, b)

    def __rtruediv__(self, other):
        if not isinstance(other, MyTensor):
            other = MyTensor(other)
        a, b = MyTensor.broadcast(self, other)
        return div(b, a)

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

    def sum(self, dim: Union[None, int, list[int]] = None):
        return sum(self, dim=dim)

    def expand(self, *, shape: list[int]):
        if tuple(self.shape) == shape:
            return self
        return expand(self, shape=shape)

    def T(self):
        return transpose(self)

    def as_column_vector(self):
        """From shape (1, n) to (n,)"""
        return as_column_vector(self)

    def as_row_vector(self):
        """From shape (n,) to (1, n)"""
        return as_row_vector(self)

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @staticmethod
    def broadcast(*tensors):
        shape = np.broadcast_shapes(*[t.shape for t in tensors])
        return tuple(t.expand(shape=shape) for t in tensors)


def _add(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value + b.value)


def _add_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs for _ in inputs)


def _add_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return grad_inputs[0] + grad_inputs[1]


def _mul(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value * b.value)


def _mul_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (inputs[1] * grad_outputs, inputs[0] * grad_outputs)


def _mul_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return inputs[1] * grad_inputs[0] + inputs[0] * grad_inputs[1]


def _matmul(a: MyTensor, b: MyTensor) -> MyTensor:
    """Matrix-matrix multiplication. For matrix-vector multiplication,
    please convert the inputs to matrices and then convert the result
    to vector manaully."""
    assert a.ndim == 2 and b.ndim == 2
    return MyTensor(a.value @ b.value)


def _matmul_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    (A, B), v = inputs, grad_outputs
    return (v @ B.T, A.T @ v)


def _matmul_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    (A, B), (dA, dB) = inputs, grad_inputs
    return dA @ B + A @ dB


def _dot(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value.dot(b.value))


def _dot_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (inputs[1] * grad_outputs, inputs[0] * grad_outputs)


def _dot_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return inputs[0].dot(grad_inputs[1]) + inputs[1].dot(grad_inputs[0])


def _sin(a: MyTensor) -> MyTensor:
    return MyTensor(np.sin(a.value))


def _sin_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs * inputs[0].cos(),)


def _sin_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return grad_inputs[0] * inputs[0].cos()


def _cos(a: MyTensor) -> MyTensor:
    return MyTensor(np.cos(a.value))


def _cos_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (-grad_outputs * inputs[0].sin(),)


def _cos_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return -grad_inputs[0] * inputs[0].sin()


def _exp(a: MyTensor) -> MyTensor:
    return MyTensor(np.exp(a.value))


def _exp_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs * outputs,)


def _exp_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return grad_inputs[0] * outputs


def _sub(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value - b.value)


def _sub_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs, -grad_outputs)


def _sub_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return grad_inputs[0] - grad_inputs[1]


def _neg(a: MyTensor) -> MyTensor:
    return MyTensor(-a.value)


def _neg_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (-grad_outputs,)


def _neg_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return -grad_inputs[0]


def _transpose(a: MyTensor) -> MyTensor:
    assert a.value.ndim == 2
    return MyTensor(a.value.T)


def _transpose_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs.T,)


def _transpose_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return grad_inputs[0].T


def _log(a: MyTensor) -> MyTensor:
    return MyTensor(np.log(a.value))


def _log_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs / inputs[0],)


def _log_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return grad_inputs[0] / inputs[0]


def _div(a: MyTensor, b: MyTensor) -> MyTensor:
    return MyTensor(a.value / b.value)


def _div_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs / inputs[1], -grad_outputs * outputs / inputs[1])


def _div_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return outputs * (grad_inputs[0] / inputs[0] - grad_inputs[1] / inputs[1])


def _sum(a: MyTensor, *, dim: Union[None, int, list[int]] = None) -> MyTensor:
    # We follow pytorch convention, where
    # torch.sum(a, dim=tuple()) is same as torch.sum(a, dim=None)
    # However, as np.sum(a, tuple()) is different from np.sum(a, None),
    # we have to convert empty list or empty tuple to None manually
    if dim is not None and not isinstance(dim, int) and len(dim) == 0:
        dim = None
    return MyTensor(np.sum(a.value, axis=dim))


def _sum_vjp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_outputs: MyTensor,
    *,
    dim: Union[None, int, list[int]] = None,
) -> list[MyTensor]:
    return (grad_outputs.expand(shape=inputs[0].shape),)


def _sum_jvp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_inputs: list[MyTensor],
    *,
    dim: Union[None, int, list[int]] = None,
) -> MyTensor:
    return grad_inputs[0].sum(dim=dim)


def _expand(a: MyTensor, *, shape: list[int]) -> MyTensor:
    return MyTensor(np.broadcast_to(a.value, shape))


def _expand_vjp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_outputs: MyTensor,
    *,
    shape: list[int],
) -> list[MyTensor]:
    inputs_shape = list(inputs[0].shape)
    # prepend to align with the required shape
    inputs_shape = [1] * (len(inputs_shape) - len(shape)) + inputs_shape
    # computes axis to be reduced, i.e., the axes where expand occurs
    dim = tuple(i for i, (a, b) in enumerate(zip(inputs_shape, shape)) if a != b)
    return (grad_outputs.sum(dim=dim).expand(shape=inputs[0].shape),)


def _expand_jvp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_inputs: list[MyTensor],
    *,
    shape: list[int],
) -> MyTensor:
    return grad_inputs[0].expand(shape=shape)


def _as_column_vector(a: MyTensor) -> MyTensor:
    # expect a is a row vector
    assert a.ndim == 2 and a.shape[0] == 1
    return MyTensor(np.reshape(a.value.copy(), -1))


def _as_column_vector_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs.as_row_vector(),)


def _as_column_vector_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return grad_inputs[0].as_column_vector()


def _as_row_vector(a: MyTensor) -> MyTensor:
    # expect a is a column vector
    assert a.ndim == 1
    return MyTensor(np.reshape(a.value.copy(), (1, -1)))


def _as_row_vector_vjp(
    inputs: list[MyTensor], outputs: MyTensor, grad_outputs: MyTensor
) -> list[MyTensor]:
    return (grad_outputs.as_column_vector(),)


def _as_row_vector_jvp(
    inputs: list[MyTensor], outputs: MyTensor, grad_inputs: list[MyTensor]
) -> MyTensor:
    return grad_inputs[0].as_row_vector()


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
sum = MyFunction("Sum", _sum, func_vjp=_sum_vjp, func_jvp=_sum_jvp)
matmul = MyFunction("Matmul", _matmul, func_vjp=_matmul_vjp, func_jvp=_matmul_jvp)
expand = MyFunction("Expand", _expand, func_vjp=_expand_vjp, func_jvp=_expand_jvp)
transpose = MyFunction(
    "Transpose", _transpose, func_vjp=_transpose_vjp, func_jvp=_transpose_jvp
)
as_column_vector = MyFunction(
    "As column vector",
    _as_column_vector,
    func_vjp=_as_column_vector_vjp,
    func_jvp=_as_column_vector_jvp,
)
as_row_vector = MyFunction(
    "As row vector",
    _as_row_vector,
    func_vjp=_as_row_vector_vjp,
    func_jvp=_as_row_vector_jvp,
)


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
    for k_inputs, k_outputs, k_phi, k_kwargs in reversed(call_tape):
        # chain rule
        grad_inputs = k_phi.vjp(k_inputs, k_outputs, k_outputs.grad, **k_kwargs)
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


    print("Test with function f(a, b) = Dot(a,a+b)")

    def test_f2(a, b):
        z = a + b
        z = a.dot(z)
        return z

    def test_f2_vjp(a, b, v):
        grad_a = v * (2 * a + b)
        grad_b = v * a
        return grad_a, grad_b

    a = MyTensor(np.random.randn(3))
    b = MyTensor(np.random.randn(3))
    v = MyTensor(np.random.randn(1))

    grad_a, grad_b = reverseAD(test_f2, [a, b], v)
    expected_grad_a, expected_grad_b = test_f2_vjp(a, b, v)
    match_a = np.allclose(grad_a.value, expected_grad_a.value)
    match_b = np.allclose(grad_b.value, expected_grad_b.value)

    print(f"Gradient of a: {grad_a}. Matches expected value: {match_a}")
    print(f"Gradient of b: {grad_b}. Matches expected value: {match_b}")


    print("Test with function f(a, k) = Dot(a,a+k1)")

    def test_f3(a, k):
        assert k.ndim == 0
        z = a + k
        z = a.dot(z)
        return z

    def test_f3_vjp(a, k, v):
        assert k.ndim == 0
        grad_a = v * (2 * a + k)
        grad_k = (v * a).sum()
        return grad_a, grad_k

    a = MyTensor(np.random.randn(3))
    k = MyTensor(np.asarray(np.random.randn()))
    v = MyTensor(np.random.randn(1))


    grad_a, grad_k = reverseAD(test_f3, [a, k], v)
    expected_grad_a, expected_grad_k = test_f3_vjp(a, k, v)
    match_a = np.allclose(grad_a.value, expected_grad_a.value)
    match_k = np.allclose(grad_k.value, expected_grad_k.value)

    print(f"Gradient of a: {grad_a}. Matches expected value: {match_a}")
    print(f"Gradient of k: {grad_k}. Matches expected value: {match_k}")


    print("Test with function f(a, b) = Dot(a,a)+Dot(a,b)-Sin(Dot(a,b))")

    def test_f4(a, b):
        z1 = a.dot(a)
        z2= a.dot(b)
        return z1 + z2 - z2.sin()

    def test_f4_vjp(a, b, v):
        grad_a = 2 * a + b - b * (a.dot(b).cos())
        grad_b = a - a * (a.dot(b).cos())
        return grad_a * v, grad_b * v

    a = MyTensor(np.random.randn(3))
    b = MyTensor(np.random.randn(3))
    v = MyTensor(np.random.randn(1))


    grad_a, grad_b = reverseAD(test_f4, [a, b], v)
    expected_grad_a, expected_grad_b = test_f4_vjp(a, b, v)
    match_a = np.allclose(grad_a.value, expected_grad_a.value)
    match_b = np.allclose(grad_b.value, expected_grad_b.value)

    print(f"Gradient of a: {grad_a}. Matches expected value: {match_a}")
    print(f"Gradient of b: {grad_b}. Matches expected value: {match_b}")


    # examine computation history
    # for call_inputs, call_output, myfunc, kwargs in my_func_tracker.call_tape:
    #     print(f"Function: {myfunc.name} (with kwargs {kwargs})")
    #     print(f"\tInputs: {call_inputs}")
    #     print(f"\tOutput: {call_output}")


if __name__ == "__main__":
    main()
