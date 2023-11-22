import tvm
from tvm import te
import numpy as np
import time

###############################################################################
# 1. Define Convolution
# In this example, we will compute the convolution which has
# 64(batch)x128(in_channel)x28x28(input_size) Input and
# 3x3(kernel_size)x128(in_channel)x256(out_channel) Weight.
# Firstly, we should define the placeholders of tensors and
# define the padding and convolution using "tensor expression".
###############################################################################

# Convolution dimensions
batch = 1
in_size = 28
in_channel = 128
out_channel = 256
kernel_size = 3
pad = 1
stride = 1

# Placeholder of input, weight tensors
X = te.placeholder((in_size, in_size, in_channel, batch), name="X")
W = te.placeholder((kernel_size, kernel_size, in_channel, out_channel), name="W")
out_size = (in_size - kernel_size + 2 * pad) // stride + 1

# Computation: Input padding
Xpad = te.compute(
    (in_size + 2 * pad, in_size + 2 * pad, in_channel, batch),
    lambda yy, xx, cc, nn: tvm.tir.if_then_else(
        tvm.tir.all(yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size),
        X[yy - pad, xx - pad, cc, nn],
        tvm.tir.const(0.0, "float32"),
    ),
    name="Xpad",
)

# Define reduction axis (input channel, kernel x, kernel y)
rc = te.reduce_axis((0, in_channel), name="rc") # input channel
ry = te.reduce_axis((0, kernel_size), name="ry") # kernel y
rx = te.reduce_axis((0, kernel_size), name="rx") # kernel x

# Computation: Convolution
Y = te.compute(
    (out_size, out_size, out_channel, batch),
    lambda yy, xx, ff, nn: te.sum(
        Xpad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff], axis=[ry, rx, rc]
    ),
    name="Y",
)

# Print out the initial schedule
print("\n")
print("-------------------------------------------------------------------------------")
print("Initial schedule")
print("-------------------------------------------------------------------------------")
s = te.create_schedule([Y.op])
print(tvm.lower(s, [X, Xpad, W, Y], simple_mode=True))

###############################################################################
# 2. Define Memory Hierarchy: 
# Define the memory hierarchy for buffers. GPU has a shared memory, which is
# programmable cache buffer. And maximizing the data reuse in the shared memory is
# critical to achieve high performance in GPU kernels.
###############################################################################

s = te.create_schedule(Y.op)
s[Xpad].compute_inline()  # compute Xpad inline
XX = s.cache_read(Xpad, "shared", [Y]) # X in shared memory
WW = s.cache_read(W, "shared", [Y]) # Y in shared memory
XL = s.cache_read(XX, "local", [Y]) # X in local register
WL = s.cache_read(WW, "local", [Y]) # X in local register
YL = s.cache_write(Y, "local")

# print(tvm.lower(s, [X, Xpad, W, YL], simple_mode=True))

###############################################################################
# 3. Tiling
###############################################################################

# tile consts
tile = 8
num_thread = 8
block_factor = tile * num_thread
step = 8
vthread = 2

# Get the GPU thread indices
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")

# Split the workloads
hi, wi, fi, ni = s[Y].op.axis
bz = s[Y].fuse(hi, wi)
by, fi = s[Y].split(fi, factor=block_factor)
bx, ni = s[Y].split(ni, factor=block_factor)

# Bind the iteration variables to GPU thread indices
s[Y].bind(bz, block_z)
s[Y].bind(by, block_y)
s[Y].bind(bx, block_x)

###############################################################################
# 4. Virtual Thread Mapping
###############################################################################

tyz, fi = s[Y].split(fi, nparts=vthread)  # virtual thread split
txz, ni = s[Y].split(ni, nparts=vthread)  # virtual thread split
ty, fi = s[Y].split(fi, nparts=num_thread)
tx, ni = s[Y].split(ni, nparts=num_thread)
s[Y].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

s[Y].bind(tyz, thread_yz)
s[Y].bind(txz, thread_xz)
s[Y].bind(ty, thread_y)
s[Y].bind(tx, thread_x)

###############################################################################
# 5. Cooperative Fetching
################################################################################

# Schedule YL local write
s[YL].compute_at(s[Y], tx)
yi, xi, fi, ni = s[YL].op.axis
ry, rx, rc = s[YL].op.reduce_axis
rco, rci = s[YL].split(rc, factor=step)
s[YL].reorder(rco, ry, rx, rci, fi, ni)

# Attach computation to iteration variables
s[XX].compute_at(s[YL], rx)
s[WW].compute_at(s[YL], rx)
s[XL].compute_at(s[YL], rci)
s[WL].compute_at(s[YL], rci)

# Schedule for X's shared memory load
yi, xi, ci, ni = s[XX].op.axis
ty, ci = s[XX].split(ci, nparts=num_thread)
tx, ni = s[XX].split(ni, nparts=num_thread)
_, ni = s[XX].split(ni, factor=4)
s[XX].reorder(ty, tx, yi, xi, ci, ni)
s[XX].bind(ty, thread_y)
s[XX].bind(tx, thread_x)
s[XX].vectorize(ni)  # vectorize memory load

# Schedule for W's shared memory load
yi, xi, ci, fi = s[WW].op.axis
ty, ci = s[WW].split(ci, nparts=num_thread)
tx, fi = s[WW].split(fi, nparts=num_thread)
_, fi = s[WW].split(fi, factor=4)
s[WW].reorder(ty, tx, yi, xi, ci, fi)
s[WW].bind(ty, thread_y)
s[WW].bind(tx, thread_x)
s[WW].vectorize(fi)  # vectorize memory load

# Print out the final schedule
print("\n")
print("-------------------------------------------------------------------------------")
print("Final schedule")
print("-------------------------------------------------------------------------------")
print(tvm.lower(s, [X, Xpad, W, YL, Y], simple_mode=True))

###############################################################################
# 6. Build convolution kernel
###############################################################################

func = tvm.build(s, [X, W, Y], "cuda")
dev = tvm.cuda(0) # GPU
a_np = np.random.uniform(size=(in_size, in_size, in_channel, batch)).astype(X.dtype)
w_np = np.random.uniform(size=(kernel_size, kernel_size, in_channel, out_channel)).astype(W.dtype)
a = tvm.nd.array(a_np, dev)
w = tvm.nd.array(w_np, dev)
b = tvm.nd.array(np.zeros((out_size, out_size, out_channel, batch), dtype=Y.dtype), dev)
func(a, w, b)

###############################################################################
# 7. Numpy convolution implementation
###############################################################################

def conv2d(input, kernel, stride, pad):
    # Padding
    input_padded = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    # Dimensions of the output
    batch_size, in_channel, in_height, in_width = input.shape
    out_channel, _, kernel_height, kernel_width = kernel.shape
    out_height = (in_height - kernel_height + 2 * pad) // stride + 1
    out_width = (in_width - kernel_width + 2 * pad) // stride + 1

    # Output tensor
    output = np.zeros((batch_size, out_channel, out_height, out_width))

    for b in range(batch_size):
        for k in range(out_channel):
            for i in range(0, out_height, stride):
                for j in range(0, out_width, stride):
                    output[b, k, i, j] = np.sum(input_padded[b, :, i:i+kernel_height, j:j+kernel_width] * kernel[k, :]) 

    return output

###############################################################################
# 8. Numpy convolution evaluation
###############################################################################
print("\n")
print("-------------------------------------------------------------------------------")
print("Performance measurement")
print("-------------------------------------------------------------------------------")
# Random input and kernel
input = np.random.rand(batch, in_channel, in_size, in_size)
kernel = np.random.rand(out_channel, in_channel, kernel_size, kernel_size)

# Time the convolution
start_time = time.time()
output = conv2d(input, kernel, stride, pad)
end_time = time.time()
print("Convolution(Numpy): %f ms" % ((end_time - start_time) * 1e3))

###############################################################################
# 8. TVM convolution evaluation
###############################################################################

evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("Convolution(TVM): %f ms" % (evaluator(a, w, b).mean * 1e3))