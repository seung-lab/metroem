import pdb
import time
from torch.cuda.amp import autocast
import torch.autograd.profiler as profiler

# functions
divider = '---------------------------------------'

def time_function(fun, include_gradients, iterations, *args, **kwargs):

    res = fun(*args, **kwargs)
    grad_tensor = torch.ones_like(res)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(iterations):
        res = fun(*args, **kwargs)
        if include_gradients:
            res.requires_grad = True
            res.backward(grad_tensor)
        torch.cuda.synchronize()
    end_time = time.time()
    print(f'{fun.__name__}: -- result data type: {res.dtype}, runtime: {end_time - start_time}')

def run_tests(fields, kernels, dtypes, include_gradients, iterations):
    for i in range(len(dtypes)):
        print(divider)
        dtype = dtypes[i]
        field = fields[i]
        kernel = kernels[i]
        print(f'data type: {dtype} ---------')

        fun = torch.sum
        time_function(fun, include_gradients, iterations, field)
        fun = torch.mul
        time_function(fun, include_gradients, iterations, field, field)
        fun = torch.pow
        time_function(fun, include_gradients, iterations, field, 2)
        fun = torch.sqrt
        time_function(fun, include_gradients, iterations, field)
        fun = torch.conv2d
        time_function(fun, include_gradients, iterations, field, kernel)

        print(divider)

# parameters
device = 'cuda'

i_batch = 8
i_channels = 4
i_height = 2048
i_width = 2048

k_output_channels = 8
k_input_channels = i_channels
k_height = 3
k_width = 3

dtypes = [torch.float32, torch.float16]

include_gradients = True

iterations = 5

# setup
fields = [torch.rand([i_batch, i_channels, i_height, i_width], dtype = dtype, device = device) for dtype in dtypes] 
kernels = [torch.rand([k_output_channels, k_input_channels, i_height, i_width], dtype = dtype, device = device) for dtype in dtypes] 


print('Starting PyTorch benchmark: ---------------------')
print(f'device: {device}')
print(f'input size: ({i_batch}, {i_channels}, {i_height}, {i_width})')
print(f'kernel size: ({k_output_channels}, {k_output_channels}, {k_height}, {k_width})')
print(f'data types: {dtypes})')
print(f'include gradients: {include_gradients}')
print(f'iterations: {iterations}')
print(divider)


# actual execution

print('--- NOT using AMP ---')
run_tests(fields, kernels, dtypes, include_gradients, iterations)
with autocast():
    print('--- Using AMP (*gradients are in AMP context*) ---')
    run_tests(fields, kernels, dtypes, include_gradients, iterations)
