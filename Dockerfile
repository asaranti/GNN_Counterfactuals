FROM debian:11

RUN apt-get update && apt-get install -y \
    supervisor nginx sudo pip wget mc

RUN pip3 install --upgrade pip

# Setup Frontend
RUN sudo apt update -qq
# install two helper packages we need
RUN sudo apt install -y --no-install-recommends software-properties-common dirmngr
RUN sudo apt install -y --no-install-recommends r-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-gnutls-dev \
    libcairo2-dev \
    libxt-dev \
    libssl-dev \
    libssh2-1-dev \
	ufw \
    && rm -rf /var/lib/apt/lists/*

RUN sudo su - -c "R -e \"install.packages('shiny', repos='http://cran.rstudio.com/')\""
RUN wget https://download3.rstudio.org/ubuntu-18.04/x86_64/shiny-server-1.5.18.987-amd64.deb
RUN sha256sum shiny-server-1.5.18.987-amd64.deb
RUN apt update
RUN sudo apt install -y gdebi-core libglpk40
RUN sudo gdebi shiny-server-1.5.18.987-amd64.deb
RUN echo "local(options(shiny.port = 3838, shiny.host = '0.0.0.0'))" > /usr/lib/R/etc/Rprofile.site
RUN sudo ufw allow 3838
RUN rm -f ubuntu1804/x86_64/shiny-server-1.5.18.987-amd64.deb

COPY ./xAI-Shiny-App /frontend
RUN R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))"
ENV RENV_VERSION 0.16.0
RUN R -e "remotes::install_github('rstudio/renv@${RENV_VERSION}')"
RUN R -e "install.packages('Matrix', repos = c(CRAN = 'https://cloud.r-project.org'))"
RUN R -e "options(renv.consent = TRUE); renv::restore(lockfile = '/frontend/R_shiny/code/renv.lock', repos = c(CRAN='https://packagemanager.rstudio.com/all/__linux__/focal/latest'))"

# make all app files readable
RUN chmod -R 755 /frontend/
RUN chmod -R 755 /usr/local/lib/R

# Setup Docker entrypoint
COPY ./GNN_Counterfactuals/server_config/supervisord.conf /supervisord.conf
COPY ./GNN_Counterfactuals/server_config/nginx /etc/nginx/sites-available/default
COPY ./GNN_Counterfactuals/server_config/docker-entrypoint.sh /entrypoint.sh

# Setup Backend
COPY ./GNN_Counterfactuals /backend
RUN chmod -R 755 /backend/models
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
    conda create -n gnn && \
    conda activate gnn

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN sudo apt-get update

RUN conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
RUN conda install pyg=2.0.4 -c pyg
RUN conda install captum -c pytorch
RUN conda install networkx
RUN conda install numpy==1.22.3 pandas==1.4.2 Flask apscheduler bokeh pytest==7.1.1
RUN rm -f ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN apt-get clean

EXPOSE 9000 9001

ENTRYPOINT ["sh", "/entrypoint.sh"]