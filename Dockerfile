ARG CADET_CORE_VERSION=5.0.3
ARG CONDA_VERSION=24.11.3
FROM ubuntu:noble-20250127 AS build

WORKDIR /cadet

USER root

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install build-essential cmake libhdf5-dev libsuperlu-dev intel-mkl git git-lfs libeigen3-dev && \
    apt-get clean

RUN git clone https://github.com/cadet/CADET-core CADET

#RUN git checkout

RUN mkdir -p CADET/build

WORKDIR CADET/build

SHELL ["/bin/bash", "-c"]

ENV MKLROOT=/opt/intel/mkl

RUN cmake -DCMAKE_INSTALL_PREFIX="../install" -DENABLE_STATIC_LINK_DEPS=ON -DENABLE_STATIC_LINK_LAPACK=ON -DBLA_VENDOR=Intel10_64lp_seq ../

RUN make -j $(lscpu | grep 'CPU(s)' | head -n 1 | cut -d ':' -f2 | tr -d ' ') install

RUN /cadet/CADET/install/bin/createLWE -o /cadet/CADET/install/bin/LWE.h5
RUN /cadet/CADET/install/bin/cadet-cli /cadet/CADET/install/bin/LWE.h5

FROM ubuntu:noble-20250127 AS deploy
COPY --from=build /cadet/CADET/install /cadet/CADET/install
COPY --from=build /usr/lib/x86_64-linux-gnu/libsz.so.2 /cadet/CADET/install/lib

RUN #apt-get update && \
#    apt-get -y install libhdf5-dev libsuperlu-dev git git-lfs && \
#    apt-get clean

WORKDIR /tmp

RUN /cadet/CADET/install/bin/cadet-cli --version
#RUN /cadet/CADET/install/bin/createLWE -o LWE.h5
#RUN /cadet/CADET/install/bin/cadet-cli LWE.h5
#
## --------------------------
## All of this can be removed once  https://github.com/conda-forge/miniforge-images/pull/145 is merged
#ARG MINIFORGE_NAME=Miniforge3
#ARG MINIFORGE_VERSION=24.11.3-2
#ARG TARGETPLATFORM
#
#ENV CONDA_DIR=/opt/conda
#ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
#ENV PATH=${CONDA_DIR}/bin:${PATH}
#
## 1. Install just enough for conda to work
## 2. Keep $HOME clean (no .wget-hsts file), since HSTS isn't useful in this context
## 3. Install miniforge from GitHub releases
## 4. Apply some cleanup tips from https://jcrist.github.io/conda-docker-tips.html
##    Particularly, we remove pyc and a files. The default install has no js, we can skip that
## 5. Activate base by default when running as any *non-root* user as well
##    Good security practice requires running most workloads as non-root
##    This makes sure any non-root users created also have base activated
##    for their interactive shells.
## 6. Activate base by default when running as root as well
##    The root user is already created, so won't pick up changes to /etc/skel
#RUN apt-get update > /dev/null && \
#    apt-get install --no-install-recommends --yes \
#        wget bzip2 ca-certificates \
#        git \
#        tini \
#        > /dev/null && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/* && \
#    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
#    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
#    rm /tmp/miniforge.sh && \
#    conda clean --tarballs --index-cache --packages --yes && \
#    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
#    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
#    conda clean --force-pkgs-dirs --all --yes  && \
#    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
#    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc
#
## Prevents Python from writing pyc files.
#ENV PYTHONDONTWRITEBYTECODE=1
#
## Keeps Python from buffering stdout and stderr to avoid situations where
## the application crashes without emitting any logs due to buffering.
#ENV PYTHONUNBUFFERED=1
###FROM cadet/cadetcore:latest-noble
### This is the default CADET-RDM Dockerfile content
#COPY environment.yml /tmp/environment.yml
#
#RUN conda env update -n base -f /tmp/environment.yml && \
#    conda clean --all --yes

