# ConvStencil

> ConvStencil: Transform Stencil Computation to Matrix Multiplication on Tensor Cores

## Abstract

This artifact contains the source code of ConvStencil, a novel stencil computing system to transform stencil computation to matrix multiplication on Tensor Cores efficiently.

## Prerequisites

- Hardware
    - x86-64 CPU
    - a single NVIDIA A100 GPU
- Software (attached in the docker image)
    - CUDA - 12.2 (Tested). Lower versions down to CUDA 11.0 are also supported, but it may affect the performance.
    - GCC - above 9.4.0. You may also try to use icx or clang.
    - cuDNN - above 8.0

## Getting Code
The code can be downloaded using git:
```
git clone https://github.com/microsoft/ConvStencil.git
```

## Compile

Use the following commands:
```
mkdir -p build
cd build
cmake ..
make all -j24
```

## Usage

You can run `convstencil` in the following input format.
```
convstencil_program shape input_size time_interation_size options
```
- `convstencil_program` can be chosen from `convstencil_1d`, `convstencil_2d`, and `convstencil_3d` for different dimensions.
- `shape` can be chosen by the different dimension:
    - `1d1r` and `1d2r` for 1D
    - `star2d1r`, `box2d1r`, `star2d3r` and `box2d3r` for 2D
    - `star3d1r` and `box3d1r` for 3D
- `input_size` depends on the number of dimensions; the number of inputs required is equal to the number of dimensions.
- `time_interation_size` is the iteration time.
- `options`:
    - `--help` prints the help information.
    - `--custom` inputs the custom stencil kernel weights.

## Contact

If you have any questions, please send an email to the author at kunli@microsoft.com.

## Reference


Yuetao Chen, Kun Li, Yuhao Wang, Donglin Bai, Lei Wang, Lingxiao Ma, Liang Yuan, Yunquan Zhang, Ting Cao, Mao Yang. [ConvStencil: Transform Stencil Computation to Matrix Multiplication on Tensor Cores](https://doi.org/10.1145/3627535.3638476). In *ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP)*, pp. 333–347, 2024.   

If you use our code, please cite our paper:
```
@inproceedings{10.1145/3627535.3638476,
author = {Chen, Yuetao and Li, Kun and Wang, Yuhao and Bai, Donglin and Wang, Lei and Ma, Lingxiao and Yuan, Liang and Zhang, Yunquan and Cao, Ting and Yang, Mao},
title = {ConvStencil: Transform Stencil Computation to Matrix Multiplication on Tensor Cores},
year = {2024},
isbn = {9798400704352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627535.3638476},
doi = {10.1145/3627535.3638476},
booktitle = {Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
pages = {333–347},
series = {PPoPP '24}
}
```

