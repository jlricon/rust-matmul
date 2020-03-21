# The Glorious Rust Matrix Multiplication Benchmark

What's the best way to multiply matrices in Rust? Here's a benchmark of a few libraries.

It should just ran, but needs some setup for torch:
For torch you'll need to do first this
```
export LIBTORCH=~/Downloads/libtorch/  
export LD_LIBRARY_PATH=${LIBTORCH}/lib 
```
Compile and run with `TORCH_CUDA_VERSION=10.2  RUSTFLAGS="-C target-cpu=native -C codegen-units=1" cargo bench`. 

Tested on a Dell XPS 15 Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz (Has up to AVX2). Admittedly this test is at least a bit biased against torch; and it would probably excel if we had a real program running more operations.

All things considered, I would recommend ndarray as THE array/linalg crate of choice for the Rust ecosystem (If one is CPU bound).
If however one is doing a lot of smaller matrix multiplications that can fit on the stack, then the statically allocated Nalgebra matrices would be the best.

## Square matrix multiplication
<img src="base_case.svg"/>

## N x 200 matrix multiplication
<img src="non_square.svg"/>
