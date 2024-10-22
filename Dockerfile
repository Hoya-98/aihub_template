FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy

ENV PATH /opt/conda/bin:$PATH

RUN conda create -y -n base python=3.11 && \
    echo "source activate base" > ~/.bashrc

RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
RUN pip install tensorflow[and-cuda]

RUN curl -fsSL https://code-server.dev/install.sh | sh
EXPOSE 8080

WORKDIR /workspace

RUN mkdir -p ~/.config/code-server/ && \
    echo "bind-addr: 0.0.0.0:8080" > ~/.config/code-server/config.yaml && \
    echo "auth: none" >> ~/.config/code-server/config.yaml

CMD ["code-server", "--bind-addr", "0.0.0.0:8080", "--auth", "none"]
