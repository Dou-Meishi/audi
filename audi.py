from collections import defaultdict
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

        self.debug = 0

    def reset(self):
        """Reset the call tape to empty."""
        self.call_tape = []

    def __call__(self, func):
        """Wrap the function to track inputs and outputs in `self.call_tape`.
        Expect func receive self as its first argument."""

        @functools.wraps(func)
        def wrapper(objself, *args, **kwargs):
            if self.debug > 1:
                print(f"Function: {objself.name} (with kwargs {kwargs})")
                print(f"\tInputs: {args}")

            if self.do_track:
                output = func(objself, *args, **kwargs)
                self.call_tape.append((args, output, objself, kwargs))
                return output
            else:
                return func(objself, *args, **kwargs)

        return wrapper

    @contextmanager
    def track_func(self, do_track: bool, tape: Union[None, list] = None):
        """Context manager to enable or disable tracking within a block.  If
        tape is not None, store records in it. Otherwise, store records in
        `self.call_tape`."""
        if tape is None:
            tape = self.call_tape  # use self.call_tape by default
        # store old attributes
        old_do_track, old_call_tape = self.do_track, self.call_tape
        try:
            # track calls and store them in tape
            self.call_tape = tape
            yield
        finally:
            # restore old attributes
            self.do_track, self.call_tape = old_do_track, old_call_tape


# initialize a global function tracker
my_func_tracker = MyFuncTracker(do_track=True)
# apply it to MyFunction class
MyFunction.__call__ = my_func_tracker(MyFunction.__call__)


class MyTensor(object):
    def __init__(self, value=0):
        self.value = np.asarray(value)
        self.buffer = defaultdict(self.default_grad)

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

    def sum(self, *, dim: Union[None, int, list[int]] = None, keepdim: bool = False):
        return sum(self, dim=dim, keepdim=keepdim)

    def expand(self, *, shape: list[int]):
        if tuple(self.shape) == shape:
            return self
        return expand(self, shape=shape)

    def squeeze(self, *, dim: Union[None, int, list[int]] = None):
        if dim is None:
            dim = tuple(idx for idx, i in enumerate(self.shape) if i == 1)
        return squeeze(self, dim=dim)

    def unsqueeze(self, *, dim: Union[int, list[int]]):
        return unsqueeze(self, dim=dim)

    @property
    def T(self):
        return transpose(self)

    def as_column_vector(self):
        """From shape (1, n) to (n,)"""
        return self.squeeze(dim=0)

    def as_row_vector(self):
        """From shape (n,) to (1, n)"""
        return self.unsqueeze(dim=0)

    def default_grad(self):
        return MyTensor(np.zeros_like(self.value))

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
    assert a.ndim == 1 and b.ndim == 1
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


def _sum(
    a: MyTensor, *, dim: Union[None, int, list[int]] = None, keepdim: bool = False
) -> MyTensor:
    # We follow pytorch convention, where
    # torch.sum(a, dim=tuple()) is same as torch.sum(a, dim=None)
    # However, as np.sum(a, tuple()) is different from np.sum(a, None),
    # we have to convert empty list or empty tuple to None manually
    if dim is not None and not isinstance(dim, int) and len(dim) == 0:
        dim = None
    return MyTensor(np.sum(a.value, axis=dim, keepdims=keepdim))


def _sum_vjp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_outputs: MyTensor,
    *,
    dim: Union[None, int, list[int]] = None,
    keepdim: bool = False,
) -> list[MyTensor]:
    if dim is None:
        dim = tuple(range(inputs[0].ndim))
    if not keepdim:
        grad_outputs = grad_outputs.unsqueeze(dim=dim)
    return (grad_outputs.expand(shape=inputs[0].shape),)


def _sum_jvp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_inputs: list[MyTensor],
    *,
    dim: Union[None, int, list[int]] = None,
    keepdim: bool = False,
) -> MyTensor:
    return grad_inputs[0].sum(dim=dim, keepdim=keepdim)


def _expand(a: MyTensor, *, shape: list[int]) -> MyTensor:
    return MyTensor(np.broadcast_to(a.value, shape))


def _expand_vjp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_outputs: MyTensor,
    *,
    shape: list[int],
) -> list[MyTensor]:
    """Reduce `grad_outputs` to `inputs[0].shape`.

    Example. For input shape (4, 1, 3, 1) and output shape (7, 1, 4, 5, 3, 6),

    1. prepend the input shape to align with the required shape

        `padded_input_shape = (1, 1, 4, 1, 3, 1)`

    2. identify axes to be reduced, i.e., the axes where expand occurs

        `dim = (0, 3, 5)`

    3. sum `grad_outputs` along `dim` with `keepdim=True`. After that,

        `grad_input.shape == padded_input_shape`

    4. squeeze prepended shape. After that

        `grad_input.shape == inputs[0].shape`
    """
    # 1. prepend the input shape to align with the required shape
    pad = len(shape) - inputs[0].ndim
    padded_input_shape = (1,) * pad + tuple(inputs[0].shape)
    # 2. identify axes to be reduced, i.e., the axes where expand occurs
    dim = tuple(i for i, (a, b) in enumerate(zip(padded_input_shape, shape)) if a != b)
    # 3. sum `grad_outputs` along `dim` with `keepdim=True`
    grad_input = grad_outputs.sum(dim=dim, keepdim=True)
    # 4. squeeze prepended shape
    if pad > 0:
        grad_input = grad_input.squeeze(dim=list(range(pad)))
    return (grad_input,)


def _expand_jvp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_inputs: list[MyTensor],
    *,
    shape: list[int],
) -> MyTensor:
    return grad_inputs[0].expand(shape=shape)


def _squeeze(a: MyTensor, *, dim: Union[int, list[int]]) -> MyTensor:
    if isinstance(dim, int):
        dim = [dim]
    return MyTensor(np.squeeze(a.value, tuple(dim)))


def _squeeze_vjp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_outputs: MyTensor,
    *,
    dim: Union[int, list[int]],
) -> list[MyTensor]:
    return (grad_outputs.unsqueeze(dim=dim),)


def _squeeze_jvp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_inputs: list[MyTensor],
    *,
    dim: Union[None, int, list[int]] = None,
) -> MyTensor:
    return grad_inputs[0].squeeze(dim=dim)


def _unsqueeze(a: MyTensor, *, dim: Union[int, list[int]]) -> MyTensor:
    return MyTensor(np.expand_dims(a.value, dim))


def _unsqueeze_vjp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_outputs: MyTensor,
    *,
    dim: Union[int, list[int]],
) -> list[MyTensor]:
    return (grad_outputs.squeeze(dim=dim),)


def _unsqueeze_jvp(
    inputs: list[MyTensor],
    outputs: MyTensor,
    grad_inputs: list[MyTensor],
    *,
    dim: Union[None, int, list[int]] = None,
) -> MyTensor:
    return grad_inputs[0].unsqueeze(dim=dim)


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
squeeze = MyFunction("Squeeze", _squeeze, func_vjp=_squeeze_vjp, func_jvp=_squeeze_jvp)
unsqueeze = MyFunction(
    "Unsqueeze", _unsqueeze, func_vjp=_unsqueeze_vjp, func_jvp=_unsqueeze_jvp
)
transpose = MyFunction(
    "Transpose", _transpose, func_vjp=_transpose_vjp, func_jvp=_transpose_jvp
)


def reverseAD(
    f: Callable[[list[MyTensor]], MyTensor],
    inputs: list[MyTensor],
    v: MyTensor,
    *,
    gradkey: str = "grad",
) -> list[MyTensor]:
    """Use reverse-mode AD to compute the vector-Jacobian product of f.
    Return the gradient of dot(f, v) evaluated at inputs.

    Args
    ----
    - `f`: The function to be differentiated.

    - `inputs`: Inputs of `f`.

    - `v`: Any tensor matches the dim of `f`.

    - `gradkey`: A string used for the dict key. For a given tensor `a`,
           the grad is stored in `a.buffer[gradkey]`.

    Note
    ----
    The gradient of tensor `a` is accumulated in `a.bffer[gradkey]`, which is
    zero by default. However, this function does not check whether it is zero or
    not. It simply accumulates all gradient in it.
    """
    tape = []
    with my_func_tracker.track_func(True, tape=tape):
        # do computations and track in tape
        y = f(*inputs)
    # backpropagate gradient starting at y
    reverseAD_along_tape(y, tape, v, gradkey=gradkey)
    return [x.buffer[gradkey] for x in inputs]


def reverseAD_along_tape(y, call_tape, v, *, gradkey):
    """Backpropagate gradient starting at y. Initially the grad of y is set to
    v.  `gradkey` is a string used for the dict key. For a given tensor `a`, the
    grad is stored in `a.buffer[gradkey]`
    """
    y.buffer[gradkey] = v
    for k_inputs, k_outputs, k_phi, k_kwargs in reversed(call_tape):
        if my_func_tracker.debug > 0:
            print(f"VJP of: {k_phi.name} (with kwargs {k_kwargs})")
            print(f"\tInputs: {k_inputs}")

        if gradkey not in k_outputs.buffer:
            # this means k_outputs.buffer[gradkey] = 0
            # however, the VJP of any function with a zero vector returns zero
            # therefore grad_inputs will contain only zero tensors
            # so there is no need to accumulate zero into x.buffer[gradkey]
            continue

        # chain rule
        grad_inputs = k_phi.vjp(
            k_inputs, k_outputs, k_outputs.buffer[gradkey], **k_kwargs
        )
        # accumulate grad
        for x, grad in zip(k_inputs, grad_inputs):
            x.buffer[gradkey] += grad


def forwardAD(
    f: Callable[[list[MyTensor]], MyTensor],
    inputs: list[MyTensor],
    inputs_v: list[MyTensor],
    *,
    gradkey: str = "grad",
) -> MyTensor:
    """Use forward-mode AD to compute the Jacobian-vector product of f.
    Return the gradient of f(dot(v, x)) evaluated at inputs.

    Args
    ----
    - `f`: The function to be differentiated.

    - `inputs`: Inputs of `f`.

    - `inputs_v`: A list of tensor matches `inputs`.

    - `gradkey`: A string used for the dict key. For a given tensor `a`,
           the grad is stored in `a.buffer[gradkey]`.
    """
    tape = []
    with my_func_tracker.track_func(True, tape=tape):
        # do computations and track in tape
        y = f(*inputs)
    # forward propagate gradient starting at inputs
    forwardAD_along_tape(inputs, tape, inputs_v, gradkey=gradkey)
    return y.buffer[gradkey]


def forwardAD_along_tape(inputs, call_tape, inputs_v, *, gradkey):
    """Forward propagate gradient starting at inputs. Initially the grad of
    inputs is set to inputs_v.  `gradkey` is a string used for the dict key. For
    a given tensor `a`, the grad is stored in `a.buffer[gradkey]` """
    for x, v in zip(inputs, inputs_v):
        x.buffer[gradkey] = v
    for k_inputs, k_outputs, k_phi, k_kwargs in call_tape:
        if my_func_tracker.debug > 0:
            print(f"JVP of: {k_phi.name} (with kwargs {k_kwargs})")
            print(f"\tInputs: {k_inputs}")

        # chain rule
        grad_inputs = [x.buffer[gradkey] for x in k_inputs]
        k_outputs.buffer[gradkey] = k_phi.jvp(
            k_inputs, k_outputs, grad_inputs, **k_kwargs
        )


def hvp_by_AD(*args, mode: str = "rr" ,**kwargs):
    """Calculate the Hessian-vector by automatic differentiation."""
    if mode == "rr":
        return hvp_by_reverse_reverseAD(*args, **kwargs)
    elif mode == "rf":
        return hvp_by_reverse_forwardAD(*args, **kwargs)
    elif mode == "fr":
        return hvp_by_forward_reverseAD(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def hvp_by_reverse_reverseAD(
    f: Callable[[list[MyTensor]], MyTensor],
    inputs: list[MyTensor],
    v_vars: list[MyTensor],
    *,
    inputs_vars: Union[None, list[MyTensor]] = None,
) -> list[MyTensor]:
    """Calculate the Hessian-vector product of function `f` using
    reverse-on-reverse mode automatic differentiation.

    Args:
        f: A function that operates on a list of MyTensor objects and returns a
            MyTensor result.

        inputs: A list of MyTensor objects representing input parameters for the
            function `f`.

        v_vars: A list of MyTensor objects corresponding to the vectors for
            which the Hessian-vector product is calculated.

        inputs_vars: An optional subset of `inputs` specifying independent
            variables in the Hessian matrix.  `inputs_vars` aligns with the number
            of tensors in `v_vars`.

    Returns:
        A list of MyTensor objects representing the Hessian-vector product.

    Note:
        `inputs_vars` should be a subset of `inputs`, specifically highlighting
        independent variables in the Hessian matrix.
    """
    if inputs_vars is None:
        inputs_vars = inputs

    tape1 = []
    with my_func_tracker.track_func(True, tape=tape1):
        # do computations and track in tape1
        y = f(*inputs)
    tape2 = []
    with my_func_tracker.track_func(True, tape=tape2):
        # compute vector product of grad and v
        reverseAD_along_tape(y, tape1, MyTensor(1.0), gradkey="rrgrad1")
        grad_inputs = [x.buffer["rrgrad1"] for x in inputs_vars]
        yy = MyTensor(0.0)
        for grad, v in zip(grad_inputs, v_vars):
            yy += sum(grad * v)
    # apply reverse-mode AD to yy
    # ATTENTION: we have to use a different gradkey to avoid modifying inputs
    #            recorded in tape 2
    reverseAD_along_tape(yy, tape1 + tape2, MyTensor(1.0), gradkey="rrgrad2")
    return [x.buffer["rrgrad2"] for x in inputs_vars]


def hvp_by_reverse_forwardAD(
    f: Callable[[list[MyTensor]], MyTensor],
    inputs: list[MyTensor],
    v_vars: list[MyTensor],
    *,
    inputs_vars: Union[None, list[MyTensor]] = None,
) -> list[MyTensor]:
    """Calculate the Hessian-vector product of function `f` using
    reverse-on-forward mode automatic differentiation.

    See hvp_by_reverse_reverseAD for explanation of arguments.
    """
    if inputs_vars is None:
        inputs_vars = inputs

    tape1 = []
    with my_func_tracker.track_func(True, tape=tape1):
        # do computations and track in tape1
        y = f(*inputs)
    tape2 = []
    with my_func_tracker.track_func(True, tape=tape2):
        # compute vector product of grad and v
        forwardAD_along_tape(inputs_vars, tape1, v_vars, gradkey="rfgrad1")
        yy = y.buffer["rfgrad1"]
    # apply reverse-mode AD to yy
    # ATTENTION: we have to use a different gradkey to avoid modifying inputs
    #            recorded in tape 2
    reverseAD_along_tape(yy, tape1 + tape2, MyTensor(1.0), gradkey="rfgrad2")
    return [x.buffer["rfgrad2"] for x in inputs_vars]


def hvp_by_forward_reverseAD(
    f: Callable[[list[MyTensor]], MyTensor],
    inputs: list[MyTensor],
    v_vars: list[MyTensor],
    *,
    inputs_vars: Union[None, list[MyTensor]] = None,
) -> list[MyTensor]:
    """Calculate the Hessian-vector product of function `f` using
    forward-on-reverse mode automatic differentiation.

    See hvp_by_reverse_reverseAD for explanation of arguments.
    """
    if inputs_vars is None:
        inputs_vars = inputs

    tape1 = []
    with my_func_tracker.track_func(True, tape=tape1):
        # do computations and track in tape1
        y = f(*inputs)

    tape2 = []
    with my_func_tracker.track_func(True, tape=tape2):
        # compute grad
        reverseAD_along_tape(y, tape1, MyTensor(1.0), gradkey="frgrad1")
        yy = [x.buffer["frgrad1"] for x in inputs_vars]
    # apply forward-mode AD to yy
    # ATTENTION: we have to use a different gradkey to avoid modifying inputs
    #            recorded in tape 2
    forwardAD_along_tape(inputs_vars, tape1 + tape2, v_vars, gradkey="frgrad2")
    return [x.buffer["frgrad2"] for x in yy]


class Test(object):

    @staticmethod
    def f1(a, b):
        z = a + b
        z = a * z
        return z

    @staticmethod
    def f1_vjp(a, b, v):
        grad_a = v * (2 * a + b)
        grad_b = v * a
        return grad_a, grad_b

    @staticmethod
    def f1_jvp(a, b, va, vb):
        grad_a = va * (2 * a + b)
        grad_b = vb * a
        return grad_a + grad_b

    @staticmethod
    def f2(a, b):
        z = a + b
        z = a.dot(z)
        return z

    @staticmethod
    def f2_vjp(a, b, v):
        grad_a = v * (2 * a + b)
        grad_b = v * a
        return grad_a, grad_b

    @staticmethod
    def f2_jvp(a, b, va, vb):
        grad_a = 2 * a + b
        grad_b = a
        return dot(grad_a, va) + dot(grad_b, vb)

    @staticmethod
    def f2_hvp(a, b, va, vb):
        return vb + 2 * va, va

    @staticmethod
    def f3(a, k):
        assert k.ndim == 0
        z = a + k
        z = a.dot(z)
        return z

    @staticmethod
    def f3_vjp(a, k, v):
        assert k.ndim == 0
        grad_a = v * (2 * a + k)
        grad_k = (v * a).sum()
        return grad_a, grad_k

    @staticmethod
    def f4(a, b):
        z1 = a.dot(a)
        z2 = a.dot(b)
        return z1 + z2 - z2.sin()

    @staticmethod
    def f4_vjp(a, b, v):
        grad_a = 2 * a + b - b * (a.dot(b).cos())
        grad_b = a - a * (a.dot(b).cos())
        return grad_a * v, grad_b * v

    @staticmethod
    def f4_jvp(a, b, va, vb):
        grad_a = 2 * a + b - b * (a.dot(b).cos())
        grad_b = a - a * (a.dot(b).cos())
        return dot(grad_a, va) + dot(grad_b, vb)

    @staticmethod
    def f4_hvp(a, b, va, vb):
        z = dot(a, b)
        sinz, cosz = sin(z), cos(z)
        hvp_a = 2 * va + sinz * b * dot(va, b) + vb - vb * cosz + sinz * b * dot(vb, a)
        hvp_b = va - va * cosz + a * sinz * dot(va, b) + a * sinz * dot(vb, a)
        return hvp_a, hvp_b

    @staticmethod
    def f4_hvp_partial(a, b, va):
        z = dot(a, b)
        hvp_a = 2 * va + sin(z) * b * dot(va, b)
        return hvp_a,

    @staticmethod
    def f5(A, x, b):
        z = A @ x - b
        return z.dot(z)

    @staticmethod
    def f5_vjp(A, x, b, v):
        z = A @ x - b
        dA = 2 * z.as_row_vector().T @ x.as_row_vector()
        dx = 2 * A.T @ z
        db = -2 * z
        return dA * v, dx * v, db * v

    @staticmethod
    def f5_hvp_partial(A, x, b, vx, vb):
        hvp_x = 2 * A.T @ (A @ vx - vb)
        hvp_b = 2 * (vb - A @ vx)
        return hvp_x, hvp_b

    @staticmethod
    def f6(a, b):
        s = 1 / (1 + exp(-a))
        nll = -sum(b * log(s) + (1 - b) * log(1 - s))
        return nll

    @staticmethod
    def f6_vjp(a, b, v):
        s = 1 / (1 + exp(-a))
        grad_a = v * (s - b)
        grad_b = v * log(1 / s - 1)
        return grad_a, grad_b

    @staticmethod
    def f6_jvp(a, b, va, vb):
        s = 1 / (1 + exp(-a))
        grad_a = s - b
        grad_b = log(1 / s - 1)
        return dot(grad_a, va) + dot(grad_b, vb)

    @staticmethod
    def f7(a):
        return dot(a, a)

    @staticmethod
    def f7_hvp(a, va):
        return (2 * va,)

    @staticmethod
    def f8(X, w, y):
        a, b = X @ w, y
        s = 1 / (1 + exp(-a))
        nll = -sum(b * log(s) + (1 - b) * log(1 - s))
        return nll

    @staticmethod
    def f8_hvp_partial(X, w, y, vw):
        x = X @ w
        s = 1 / (1 + exp(-x))
        Omega = s * (1 - s)
        v = X @ vw
        return X.T @ (Omega * v),

    @staticmethod
    def test_f1_vjp():
        print("\nTest with function f(a, b) = a*(a+b)")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        v = MyTensor(np.random.randn(3))

        grad_a, grad_b = reverseAD(Test.f1, [a, b], v)
        expected_grad_a, expected_grad_b = Test.f1_vjp(a, b, v)
        match_a = np.allclose(grad_a.value, expected_grad_a.value)
        match_b = np.allclose(grad_b.value, expected_grad_b.value)

        print(f"Gradient of a: {grad_a}. Matches expected value: {match_a}")
        print(f"Gradient of b: {grad_b}. Matches expected value: {match_b}")

    @staticmethod
    def test_f2_vjp():
        print("\nTest with function f(a, b) = Dot(a,a+b)")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        v = MyTensor(np.asarray(np.random.randn()))

        grad_a, grad_b = reverseAD(Test.f2, [a, b], v)
        expected_grad_a, expected_grad_b = Test.f2_vjp(a, b, v)
        match_a = np.allclose(grad_a.value, expected_grad_a.value)
        match_b = np.allclose(grad_b.value, expected_grad_b.value)

        print(f"Gradient of a: {grad_a}. Matches expected value: {match_a}")
        print(f"Gradient of b: {grad_b}. Matches expected value: {match_b}")

    @staticmethod
    def test_f3_vjp():
        print("\nTest with function f(a, k) = Dot(a,a+k1)")

        a = MyTensor(np.random.randn(3))
        k = MyTensor(np.asarray(np.random.randn()))
        v = MyTensor(np.asarray(np.random.randn()))

        grad_a, grad_k = reverseAD(Test.f3, [a, k], v)
        expected_grad_a, expected_grad_k = Test.f3_vjp(a, k, v)
        match_a = np.allclose(grad_a.value, expected_grad_a.value)
        match_k = np.allclose(grad_k.value, expected_grad_k.value)

        print(f"Gradient of a: {grad_a}. Matches expected value: {match_a}")
        print(f"Gradient of k: {grad_k}. Matches expected value: {match_k}")

    @staticmethod
    def test_f4_vjp():
        print("\nTest with function f(a, b) = Dot(a,a)+Dot(a,b)-Sin(Dot(a,b))")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        v = MyTensor(np.asarray(np.random.randn()))

        grad_a, grad_b = reverseAD(Test.f4, [a, b], v)
        expected_grad_a, expected_grad_b = Test.f4_vjp(a, b, v)
        match_a = np.allclose(grad_a.value, expected_grad_a.value)
        match_b = np.allclose(grad_b.value, expected_grad_b.value)

        print(f"Gradient of a: {grad_a}. Matches expected value: {match_a}")
        print(f"Gradient of b: {grad_b}. Matches expected value: {match_b}")

    @staticmethod
    def test_f5_vjp():
        print("\nTest with function f(A, x, b) = Dot(Ax-b, Ax-b)")

        A = MyTensor(np.random.randn(4, 3))
        x = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(4))
        v = MyTensor(np.asarray(np.random.randn()))

        grad_A, grad_x, grad_b = reverseAD(Test.f5, [A, x, b], v)
        expected_grad_A, expected_grad_x, expected_grad_b = Test.f5_vjp(A, x, b, v)
        match_A = np.allclose(grad_A.value, expected_grad_A.value)
        match_x = np.allclose(grad_x.value, expected_grad_x.value)
        match_b = np.allclose(grad_b.value, expected_grad_b.value)

        print(f"Gradient of A: {grad_A}. Matches expected value: {match_A}")
        print(f"Gradient of x: {grad_x}. Matches expected value: {match_x}")
        print(f"Gradient of b: {grad_b}. Matches expected value: {match_b}")

    @staticmethod
    def test_f6_vjp():
        print("\nTest with function f(a, b) = BCEWithLogits(a, b)")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        v = MyTensor(np.asarray(np.random.randn()))

        grad_a, grad_b = reverseAD(Test.f6, [a, b], v)
        expected_grad_a, expected_grad_b = Test.f6_vjp(a, b, v)
        match_a = np.allclose(grad_a.value, expected_grad_a.value)
        match_b = np.allclose(grad_b.value, expected_grad_b.value)

        print(f"Gradient of a: {grad_a}. Matches expected value: {match_a}")
        print(f"Gradient of b: {grad_b}. Matches expected value: {match_b}")

    @staticmethod
    def test_f1_jvp():
        print("\nTest with function f(a, b) = a*(a+b)")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        va = MyTensor(np.random.randn(3))
        vb = MyTensor(np.random.randn(3))

        grad_L = forwardAD(Test.f1, [a, b], [va, vb])
        expected_grad_L = Test.f1_jvp(a, b, va, vb)
        match_L = np.allclose(grad_L.value, expected_grad_L.value)

        print(f"Gradient of f: {grad_L}. Matches expected value: {match_L}")

    @staticmethod
    def test_f2_jvp():
        print("\nTest with function f(a, b) = Dot(a,a+b)")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        va = MyTensor(np.random.randn(3))
        vb = MyTensor(np.random.randn(3))

        grad_L = forwardAD(Test.f2, [a, b], [va, vb])
        expected_grad_L = Test.f2_jvp(a, b, va, vb)
        match_L = np.allclose(grad_L.value, expected_grad_L.value)

        print(f"Gradient of f: {grad_L}. Matches expected value: {match_L}")

    @staticmethod
    def test_f4_jvp():
        print("\nTest with function f(a, b) = Dot(a,a)+Dot(a,b)-Sin(Dot(a,b))")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        va = MyTensor(np.random.randn(3))
        vb = MyTensor(np.random.randn(3))

        grad_L = forwardAD(Test.f4, [a, b], [va, vb])
        expected_grad_L = Test.f4_jvp(a, b, va, vb)
        match_L = np.allclose(grad_L.value, expected_grad_L.value)

        print(f"Gradient of f: {grad_L}. Matches expected value: {match_L}")

    @staticmethod
    def test_f6_jvp():
        print("\nTest with function f(a, b) = BCEWithLogits(a,b)")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        va = MyTensor(np.random.randn(3))
        vb = MyTensor(np.random.randn(3))

        grad_L = forwardAD(Test.f6, [a, b], [va, vb])
        expected_grad_L = Test.f6_jvp(a, b, va, vb)
        match_L = np.allclose(grad_L.value, expected_grad_L.value)

        print(f"Gradient of f: {grad_L}. Matches expected value: {match_L}")

    @staticmethod
    def test_f7_hvp(mode):
        print("\nTest with function f(a, b) = Dot(a,a)")

        a = MyTensor(np.random.randn(3))
        va = MyTensor(np.random.randn(3))

        (hvp_a,) = hvp_by_AD(Test.f7, [a], [va], mode=mode)
        (expected_hvp_a,) = Test.f7_hvp(a, va)
        match_hvp_a = np.allclose(hvp_a.value, expected_hvp_a.value)

        print(f"HVP of a: {hvp_a}. Matches expected value: {match_hvp_a}")

    @staticmethod
    def test_f2_hvp(mode):
        print("\nTest with function f(a, b) = Dot(a,a+b)")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        va = MyTensor(np.random.randn(3))
        vb = MyTensor(np.random.randn(3))

        hvp_a, hvp_b = hvp_by_AD(Test.f2, [a, b], [va, vb], mode=mode)
        expected_hvp_a, expected_hvp_b = Test.f2_hvp(a, b, va, vb)
        match_hvp_a = np.allclose(hvp_a.value, expected_hvp_a.value)
        match_hvp_b = np.allclose(hvp_b.value, expected_hvp_b.value)

        print(f"HVP of a: {hvp_a}. Matches expected value: {match_hvp_a}")
        print(f"HVP of b: {hvp_b}. Matches expected value: {match_hvp_b}")

    @staticmethod
    def test_f4_hvp(mode):
        print("\nTest with function f(a, b) = Dot(a,a+b)-Sin(Dot(a,b))")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        va = MyTensor(np.random.randn(3))
        vb = MyTensor(np.random.randn(3))

        hvp_a, hvp_b = hvp_by_AD(Test.f4, [a, b], [va, vb], mode=mode)
        expected_hvp_a, expected_hvp_b = Test.f4_hvp(a, b, va, vb)
        match_hvp_a = np.allclose(hvp_a.value, expected_hvp_a.value)
        match_hvp_b = np.allclose(hvp_b.value, expected_hvp_b.value)

        print(f"HVP of a: {hvp_a}. Matches expected value: {match_hvp_a}")
        print(f"HVP of b: {hvp_b}. Matches expected value: {match_hvp_b}")

    @staticmethod
    def test_f4_hvp_partial(mode):
        print("\nTest with function f(a) = Dot(a,a+b)-Sin(Dot(a,b))")
        print("\twhere b is a constant.")

        a = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(3))
        va = MyTensor(np.random.randn(3))

        hvp_a, = hvp_by_AD(Test.f4, [a, b], [va], inputs_vars=[a], mode=mode)
        expected_hvp_a, = Test.f4_hvp_partial(a, b, va)
        match_hvp_a = np.allclose(hvp_a.value, expected_hvp_a.value)

        print(f"HVP of a: {hvp_a}. Matches expected value: {match_hvp_a}")

    @staticmethod
    def test_f5_hvp_partial(mode):
        print("\nTest with function f(x, b) = Dot(Ax-b, Ax-b)")
        print("\twhere A is a constant")

        A = MyTensor(np.random.randn(4, 3))
        x = MyTensor(np.random.randn(3))
        b = MyTensor(np.random.randn(4))
        vx = MyTensor(np.random.randn(3))
        vb = MyTensor(np.random.randn(4))

        hvp_x, hvp_b = hvp_by_AD(Test.f5, [A, x, b], [vx, vb], inputs_vars=[x, b], mode=mode)
        expected_hvp_x, expected_hvp_b = Test.f5_hvp_partial(A, x, b, vx, vb)
        match_hvp_x = np.allclose(hvp_x.value, expected_hvp_x.value)
        match_hvp_b = np.allclose(hvp_b.value, expected_hvp_b.value)

        print(f"HVP of a: {hvp_x}. Matches expected value: {match_hvp_x}")
        print(f"HVP of a: {hvp_b}. Matches expected value: {match_hvp_b}")

    @staticmethod
    def test_f8_hvp_partial(mode):
        print("\nTest with function f(w) = BCEWithLogits(Xw, y)")
        print("\twhere X and y are constants")

        X = MyTensor(np.random.randn(3, 4))
        w = MyTensor(np.random.randn(4))
        y = MyTensor(np.random.randn(3))
        vw = MyTensor(np.random.randn(4))

        hvp_w, = hvp_by_AD(Test.f8, [X, w, y], [vw], inputs_vars=[w], mode=mode)
        expected_hvp_w, = Test.f8_hvp_partial(X, w, y, vw)
        match_hvp_w = np.allclose(hvp_w.value, expected_hvp_w.value)

        print(f"HVP of w: {hvp_w}. Matches expected value: {match_hvp_w}")


def main():
    print("\n==================================================")
    print("Test grad (with reverse-mode AD).")
    Test.test_f1_vjp()
    Test.test_f2_vjp()
    Test.test_f3_vjp()
    Test.test_f4_vjp()
    Test.test_f5_vjp()
    Test.test_f6_vjp()

    print("\n==================================================")
    print("Test grad (with forward-mode AD).")
    Test.test_f1_jvp()
    Test.test_f2_jvp()
    Test.test_f4_jvp()
    Test.test_f6_jvp()

    print("\n==================================================")
    print("Test Hessian (with reverse-on-reverse AD).")
    Test.test_f7_hvp("rr")
    Test.test_f2_hvp("rr")
    Test.test_f4_hvp("rr")
    Test.test_f4_hvp_partial("rr")
    Test.test_f5_hvp_partial("rr")
    Test.test_f8_hvp_partial("rr")

    print("\n==================================================")
    print("Test Hessian (with reverse-on-forward AD).")
    Test.test_f7_hvp("rf")
    Test.test_f2_hvp("rf")
    Test.test_f4_hvp("rf")
    Test.test_f4_hvp_partial("rf")
    Test.test_f5_hvp_partial("rf")
    Test.test_f8_hvp_partial("rf")

    print("\n==================================================")
    print("Test Hessian (with forward-on-reverse AD).")
    Test.test_f7_hvp("fr")
    Test.test_f2_hvp("fr")
    Test.test_f4_hvp("fr")
    Test.test_f4_hvp_partial("fr")
    Test.test_f5_hvp_partial("fr")
    Test.test_f8_hvp_partial("fr")


if __name__ == "__main__":
    main()
