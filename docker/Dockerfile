FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
 
RUN apt-get update && apt-get install -y \
    python-is-python3 \
    python3-dev \
    python3-pip \
    libjpeg-dev \
    zlib1g-dev \
    git \
    wget \
    unzip \
    libx11-6 \
    libgl1 \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
 
# Copy the TRELLIS script
COPY setup.sh /tmp/setup.sh

RUN pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

RUN bash /tmp/setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --nvdiffrast

RUN pip install tensorboard pandas lpips
RUN pip uninstall -y pillow
RUN pip install pillow-simd
RUN pip install transformers==4.49.0
RUN pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html
RUN pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124
RUN pip install spconv-cu118==2.3.8
ENV CUDA_HOME=/usr/local/cuda-12.4/
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git


# For A6000 and H100 GPUs
# If you have a different GPU, change the CUDA_ARCH_LIST accordingly
ENV TORCH_CUDA_ARCH_LIST="8.6;9.0"
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
RUN mkdir -p /tmp/extensions \
    # Clone the repository with submodules
    && git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast \
    # Install the package using pip
    && pip install /tmp/extensions/diffoctreerast

RUN pip install flash-attn==2.5.6 --no-build-isolation

RUN rm /tmp/setup.sh
