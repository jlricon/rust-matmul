# The Glorious Rust Matrix Multiplication Benchmark

What's the best way to multiply matrices in Rust? Here's a benchmark of a few libraries.

It should just ran, but needs some setup for torch:
For torch you'll need to do first this
```
export LIBTORCH=~/Downloads/libtorch/  
export LD_LIBRARY_PATH=${LIBTORCH}/lib 
```
Compile and run with `TORCH_CUDA_VERSION=10.2  RUSTFLAGS="-C target-cpu=native -C codegen-units=1" cargo bench`. 

Tested on a Dell XPS 15 Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz (Has up to AVX2)
## Square matrix multiplication
<img src="base_case.svg"/>
## 200 x N matrix multiplication
<img src="non_square.svg"/>
