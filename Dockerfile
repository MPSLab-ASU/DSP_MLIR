# Use an official Ubuntu 22.04 as a base
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    cmake \
    ninja-build \
    clang \
    build-essential \
    python3 \
    libedit2 \
    libedit-dev \
    libncurses6 \
    libtinfo5 \
    libncurses-dev \
    python3-pip \
    lld \
    wget \
    curl \
    vim \
    desktop-file-utils \
    gawk \
    sudo \
    systemd \
    xdg-utils && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*


# Symlinks for older libraries
RUN ln -s /usr/lib/x86_64-linux-gnu/libedit.so.2 /usr/lib/x86_64-linux-gnu/libedit.so.0 && \
    ln -s /usr/lib/x86_64-linux-gnu/libncurses.so.6 /usr/lib/x86_64-linux-gnu/libncurses.so.5



# Set environment variables
ENV PATH=/usr/local/bin:$PATH

# Clone your project (checkout docker branch directly)
RUN git clone https://github.com/MPSLab-ASU/DSP_MLIR.git /DSP_MLIR

# Set working directory
WORKDIR /DSP_MLIR
RUN git checkout docker
# Build
RUN mkdir build && cd build && \
    cmake -G Ninja ../llvm \
      -DLLVM_ENABLE_PROJECTS="mlir;clang" \
      -DLLVM_BUILD_EXAMPLES=ON \
      -DLLVM_TARGETS_TO_BUILD="Native;Hexagon" \
      -DCMAKE_BUILD_TYPE=Release && \
    ninja

# Copy additional files
# COPY hexagon_target/ /DSP_MLIR/build/bin/hexagon/
# COPY Hexagon_Tools/ /DSP_MLIR/Hexagon_Tools/

# Default to bash
CMD ["/bin/bash"]
