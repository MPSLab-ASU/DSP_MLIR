# DSP-MLIR Compiler

This repository contains the source code for **DSP-MLIR**, a compiler tailored for Digital Signal Processing (DSP) applications. It provides highly optimized tools and environments for building, optimizing, and running DSP operations like Fast Fourier Transforms (FFT), Finite Impulse Response (FIR) filters, and more.

The project is built on top of the **LLVM** infrastructure and leverages the **MLIR** (Multi-Level Intermediate Representation) framework for implementing DSP-specific operations and transformations.




## Build Instructions

To build the DSP-MLIR compiler, follow these steps:

### Step 1: Clone this repository and cd into the DSP-MLIR folder.


### Step 2: Make and cd into the build directory using the following command:

```bash
mkdir build
cd build

```
### Step 3: To build the project, run the following command:
```bash
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
```

### Step 4: After configuring the build, compile the project by running:
```bash
ninja
```

## Running an Example

After the build completes, you can run an example to test the DSP operations. From the build directory:

```bash
ninja && ./bin/dsp1 ../mlir/test/Examples/DspExample/dsp_gain_op.py -emit=mlir-affine
ninja && ./bin/dsp1 ../mlir/test/Examples/DspExample/dsp_gain_op.py -emit=jit
```

