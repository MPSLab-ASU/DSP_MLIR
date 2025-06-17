# DSP-MLIR Compiler

This repository contains the source code for **DSP-MLIR**, a compiler tailored for Digital Signal Processing (DSP) applications. It provides highly optimized tools and environments for building, optimizing, and running DSP operations like Fast Fourier Transforms (FFT), Finite Impulse Response (FIR) filters, and more.

The project is built on top of the **LLVM** infrastructure and leverages the **MLIR** (Multi-Level Intermediate Representation) framework for implementing DSP-specific operations and transformations.



# DSP-MLIR Artifact Guide

The `artifact.zip` is available on OneDrive at the following URL:  [Download the artifact here](https://arizonastateu-my.sharepoint.com/:u:/g/personal/apkhedka_sundevils_asu_edu/Edg1WrP_yUhAgQCZ5cwOV2MByjAjPSx_NyHyDxpMj4o3Lg?e=Lb5qyU)

For detailed instructions, please see the rest of this README file.


# Getting Started Guide

This guide describes how to set up and prepare the environment for reproducing the artifact. We provide two installation methods for DSP-MLIR: option 1 with Docker and option 2 with manual installation from GitHub. **We recommend using option 1** to minimize the setup effort, as building an LLVM project can take a long time.

---

## Option 1: Using Prebuilt Docker Image (Recommended)

### Prerequisites:
- Ubuntu system with Docker installed.

### Step 1: Install Docker
```bash
sudo snap install docker
```

### Step 2: Pull the prebuilt Docker image from DockerHub
```bash
docker pull apk23/dsp-mlir:latest
```
*Note: This Docker image is built using the Dockerfile available on GitHub: https://github.com/MPSLab-ASU/DSP_MLIR.git (branch: "docker")*

### Step 3: Start the Docker container with privileged mode
```bash
docker run --privileged -it apk23/dsp-mlir:latest
```

### Step 4: Navigate to the benchmark directory inside the container
```bash
cd /DSP_MLIR/mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/DSP-DSL
```

### Step 5: Run a sample experiment to test the environment
```bash
python ResultScript.py spectralAnalysis.py spectralAnalysis
```

---

## Setting up Hexagon Simulator (for Hexagon Experiments)

**Required only if you wish to evaluate Hexagon target results.**

### Step 6: Install Hexagon SDK inside the container

1. Create a Qualcomm account: https://myaccount.qualcomm.com/signup

2. Install QSC and QPM:
```bash
cd /DSP_MLIR/
mkdir Qualcomm
cd Qualcomm
wget https://softwarecenter.qualcomm.com/api/download/software/qsc/linux/latest.deb -O qsc_installer.deb
apt install ./qsc_installer.deb
```
(Press 2 and Enter to accept the license.)

3. Login to QPM:
```bash
qpm-cli --login
```
*(Use the same email and password used to create the Qualcomm account)*

4. Activate Hexagon SDK license:
```bash
qpm-cli --license-activate hexagonsdk6.x
```

5. Install Hexagon SDK 6.2.0.1:
```bash
qpm-cli --install hexagonsdk6.x --version=6.2.0.1
```
(Press "y" and Enter)

6. Install required libraries:
```bash
sudo apt-get install libncurses5
```

### Step 7: Copy Hexagon target files into the build directory
```bash
cp -r /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/Tools/target/hexagon /DSP_MLIR/build/bin/
mkdir /DSP_MLIR/Hexagon_Tools
cp -r /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/Tools/* /DSP_MLIR/Hexagon_Tools/
```

---

## Option 2: Local Setup (Manual Installation)

### Step 1: Clone the DSP-MLIR repository
```bash
git clone https://github.com/MPSLab-ASU/DSP_MLIR
```

### Step 2: Checkout the docker branch
```bash
cd DSP_MLIR
git checkout docker
```

### Step 3: Build the project
```bash
mkdir build
cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir;clang" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="Native;Hexagon" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF
ninja
```

### Step 4: Setup Hexagon SDK
Follow **Steps 6–7** in Option 1. Replace `/DSP_MLIR` with your local directory path.

---

## Important Notes

- Hexagon tools (SDK and Simulator) **are not included** in the Docker image due to licensing.
- Users must install Hexagon SDK manually.

---

## Step-by-Step Instructions for Running Experiments

---

### Step 1.1: Run CPU Benchmarks (Affine and DSP-MLIR)
```bash
cd /DSP_MLIR/mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/DSP-DSL
python RunResults.py
```
*Execution time: 2–6 hours*

**Or run individual benchmarks:**
```bash
python ResultScript.py <applicationname> <logfoldername>
```

**Examples:**
```bash
python ResultScript.py radarSignalProcessing.py radarSignalProcessing
python ResultScript.py spectralAnalysis.py spectralAnalysis
```

---

### Step 1.2: Run CPU Benchmarks (GCC and Clang)
```bash
cd /DSP_MLIR/mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/CCode
python RunResults.py
```

**Or run individual benchmarks:**
```bash
python ResultScript.py radarSignalProcessing.py radarSignalProcessing
python ResultScript.py spectralAnalysis.py spectralAnalysis
```

---

### Step 2: Analyze CPU Benchmark Results

Compare outputs with `x86_EXE` sheet in `expectedresults.xlsx`.

**Example normalization:**
- Affine: 0.0714s → normalized to 1
- DSP-MLIR: 0.0682s → 0.0682 / 0.0714 ≈ **0.955**

---

### Step 3.1: Run Hexagon Simulator Benchmarks (Affine and DSP-MLIR)
```bash
cd /DSP_MLIR/mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/DSP-DSL
python RunHexagon.py
```

**Or run individual simulations:**
```bash
python HexagonClangResultScript.py <applicationname> <logfoldername>
python HexagonResultScript.py <applicationname> <logfoldername>
```

---

### Step 3.2: Run Hexagon Simulator Benchmarks (Clang and Clang-hexagon)
```bash
cd /DSP_MLIR/mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/CCode
python RunHexagon.py
```

**Or run individual simulations:**
```bash
python HexagonClangResultScript.py spectralAnalysis.py spectralAnalysis
```

---

### Step 4: Analyze Hexagon Simulation Results

Compare results with `HEXAGON_EXE` sheet in `expectedresults.xlsx`.

**Example normalization:**
- Affine: 0.0714s → normalized to 1
- DSP-MLIR: 0.0682s → 0.0682 / 0.0714 ≈ **0.955**

---

### Step 5: Verify Programmer Productivity Improvement (LOC Reduction)

```bash
cd /DSP_MLIR/mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/
pip install pandas
python CountLinesFile.py
```

Generates:
```text
Output: consolidated_lines_of_code.csv
Path: /DSP_MLIR/mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/Output/
```

**Compare with `LinesofCode` sheet in `expectedresults.xlsx`.**

---

## Known Issues

- Full benchmark execution time: 2–6 hours.
- Minor variations in timing expected across systems.
- Relative performance trends should remain consistent.

---

## License Notice

- **Hexagon SDK** and associated tools are proprietary to **Qualcomm Technologies, Inc.**
- Not included in the Docker image or repository.
- Users must **obtain licenses** independently.

---


