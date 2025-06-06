FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    libx11-6 \
    libgl1 \
&& apt-get clean && rm -rf /var/lib/apt/lists/*
 
RUN pip install --upgrade pip
 
# Copy the TRELLIS script
COPY setup.sh /tmp/setup.sh

RUN bash /tmp/setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --nvdiffrast

RUN pip install transformers==4.49.0
RUN pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html
RUN pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124
RUN pip install spconv-cu118==2.3.8
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git
RUN pip install git+https://github.com/JeffreyXiang/diffoctreerast.git
RUN pip install git+https://github.com/autonomousvision/mip-splatting.git
 
RUN rm /tmp/setup.sh
