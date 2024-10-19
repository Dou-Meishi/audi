class MyFuncTracker(object):
    """A decorator class to track function inputs and outputs.

    Store recorded calls in attribute `call_tape`, a list of tuples
    look like (inputs_k, outputs_k, func_k)."""
    def __init__(self, do_track: bool):
        self.do_track = do_track
        self.reset()

    def reset(self):
        self.call_tape = []

    def __call__(self, func):
        """Wrap the function. Track inputs and outputs in `self.call_tape`."""
        def wrapper(*args, **kwargs):
            if self.do_track:
                inputs = args
                output = func(*args, **kwargs)
                self.call_tape.append((inputs, output, func.__name__))
                return output
            else:
                return func(*args, **kwargs)
        return wrapper

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


x = MyTensor(1.)
y = MyTensor(2.)

my_func_tracker.reset()
my_func_tracker.do_track = True
# do computations and track
z = x + y
z = x * z
my_func_tracker.do_track = False

# extract computation history
for call_inputs, call_output, func_name in my_func_tracker.call_tape:
    print(f"Function: {func_name}, Inputs: {call_inputs}, Output: {call_output}")