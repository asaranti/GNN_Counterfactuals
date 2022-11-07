FROM python:3.6-buster

RUN apt-get update && apt-get install -y \
    supervisor nginx

# Setup Docker entrypoint
COPY server_config/supervisord.conf /supervisord.conf
COPY server_config/nginx /etc/nginx/sites-available/default
COPY server_config/docker-entrypoint.sh /entrypoint.sh

# Setup backend dependencies
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    sudo \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n python-app && \
    conda activate python-app && \
    conda install python=3.6 pip

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
# RUN sudo apt-get update

RUN conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
RUN conda install pyg -c pyg
RUN conda install captum -c pytorch
RUN conda install networkx
RUN echo 'conda activate gnn \n\
alias python-app="python app.py"' >> /root/.bashrc

COPY . /backend

# Shiny R installation and app copy


EXPOSE 9000 9001

ENTRYPOINT ["sh", "/entrypoint.sh"]