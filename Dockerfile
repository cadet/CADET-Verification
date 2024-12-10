ARG MAMBA_VERSION=1.5.8
FROM mambaorg/micromamba:${MAMBA_VERSION}-noble AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

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

RUN cmake -DCMAKE_INSTALL_PREFIX="../install" -DBLA_VENDOR=Intel10_64lp_seq ../

RUN make -j $(lscpu | grep 'CPU(s)' | head -n 1 | cut -d ':' -f2 | tr -d ' ') install

USER $MAMBA_USER

WORKDIR /tmp

RUN /cadet/CADET/install/bin/cadet-cli --version
RUN /cadet/CADET/install/bin/createLWE -o LWE.h5
RUN /cadet/CADET/install/bin/cadet-cli LWE.h5

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

RUN git config --global --add safe.directory /workingdir && \
    git config --global --add safe.directory /workingdir/output && \
    git config --global user.name "IBG RDM DOCKER_USER" && \
    git config --global user.email "cadet@fz-juelich.de"

RUN git clone https://github.com/cadet/CADET-Verification.git /tmp/CADET-Verification
WORKDIR CADET-Verification
RUN git clone https://github.com/cadet/CADET-Verification-Output.git /tmp/CADET-Verification/output

