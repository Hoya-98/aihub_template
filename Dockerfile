FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    build-essential \
    bash \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    /bin/sh /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh && \
    /opt/conda/bin/conda clean -y -all

ENV PATH /opt/conda/bin:$PATH

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN conda update -n base -c defaults conda

RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
RUN pip install tensorflow[and-cuda]

RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN code-server --install-extension vscodevim.vim
RUN code-server --install-extension ms-python.python
RUN code-server --install-extension ms-toolsai.jupyter
RUN code-server --install-extension PKief.material-icon-theme
RUN code-server --install-extension GitHub.github-vscode-theme

EXPOSE 8080

WORKDIR /workspace
RUN chmod -R 777 /workspace

COPY code-server.crt /etc/ssl/certs/code-server.crt
COPY code-server.key /etc/ssl/private/code-server.key
COPY code-server.json /root/.local/share/code-server/User/settings.json

RUN mkdir -p ~/.config/code-server/ && \
    echo "bind-addr: 0.0.0.0:8080" > ~/.config/code-server/config.yaml && \
    echo "cert: /etc/ssl/certs/code-server.crt" >> ~/.config/code-server/config.yaml && \
    echo "cert-key: /etc/ssl/private/code-server.key" >> ~/.config/code-server/config.yaml && \
    echo "auth: none" >> ~/.config/code-server/config.yaml

COPY src /tmp/src

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV WORKSPACE_ENV=prod

CMD ["/usr/local/bin/entrypoint.sh"]

