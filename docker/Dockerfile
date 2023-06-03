# FROM nvidia/opengl:1.2-glvnd-devel-ubuntu20.04
# SHELL ["/bin/bash", "-ci"]

# # Timezone Configuration
# ENV TZ=Europe/Moscow
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# ENV DEBIAN_FRONTEND=noninteractive

# # avoid interactive configuration dialog from tzdata, which gets pulled in
# # as a dependency
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && \
#     apt-get install -y \
#         build-essential \
#         castxml \
#         cmake \
#         freeglut3-dev \
#         git \
#         libassimp-dev \
#         libboost-filesystem-dev \
#         libboost-numpy-dev \
#         libboost-program-options-dev \
#         libboost-python-dev \
#         libboost-serialization-dev \
#         libboost-system-dev \
#         libboost-test-dev \
#         libccd-dev \
#         libeigen3-dev \
#         libexpat1 \
#         libfcl-dev \
#         libflann-dev \
#         libtriangle-dev \
#         ninja-build \
#         pkg-config \
#         python3-celery \
#         python3-dev \
#         python3-flask \
#         python3-numpy \
#         python3-opengl \
#         python3-pip \
#         python3-pyqt5.qtopengl \
#         pypy3 \
#         wget \
#         git && \
#     # Install spot
#     wget -O /etc/apt/trusted.gpg.d/lrde.gpg https://www.lrde.epita.fr/repo/debian.gpg && \
#     echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list && \
#     apt-get update && \
#     apt-get install -y libspot-dev && \
#     # see https://github.com/mcfletch/pyopengl/issues/74
#     pip3 install pygccxml pyplusplus git+https://github.com/mcfletch/pyopengl.git@227f9c66976d9f5dadf62b9a97e6beaec84831ca#subdirectory=accelerate
# RUN wget -O - https://github.com/ompl/ompl/archive/1.6.0.tar.gz | tar zxf - && \
#     mv /ompl-1.6.0 /ompl && mkdir /ompl/build
# WORKDIR /ompl/build
# RUN cmake \
#         -DPYTHON_EXEC=/usr/bin/python3 \
#         -G Ninja \
#         /ompl && \
#     ninja update_bindings -j `nproc` && \
#     ninja -j `nproc` && \
#     ninja install


FROM ubuntu:jammy AS builder
# avoid interactive configuration dialog from tzdata, which gets pulled in
# as a dependency
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        castxml \
        cmake \
        libboost-filesystem-dev \
        libboost-numpy-dev \
        libboost-program-options-dev \
        libboost-python-dev \
        libboost-serialization-dev \
        libboost-system-dev \
        libboost-test-dev \
        libeigen3-dev \
        libexpat1 \
        libflann-dev \
        libtriangle-dev \
        ninja-build \
        pkg-config \
        python3-dev \
        python3-numpy \
        python3-pip \
        pypy3 \
        wget \
        git && \
    # Install spot
    wget -O /etc/apt/trusted.gpg.d/lrde.gpg https://www.lrde.epita.fr/repo/debian.gpg && \
    echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y libspot-dev && \
    pip3 install pygccxml pyplusplus
RUN git clone https://github.com/ompl/ompl.git
WORKDIR /build
RUN cmake \
        -DPYTHON_EXEC=/usr/bin/python3 \
        -DOMPL_REGISTRATION=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -G Ninja \
        /ompl && \
    ninja update_bindings -j `nproc` && \
    ninja -j `nproc` && \
    ninja install

FROM nvidia/opengl:1.2-glvnd-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        libboost-filesystem-dev \
        libboost-numpy-dev \
        libboost-program-options-dev \
        libboost-python-dev \
        libboost-serialization-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libtriangle-dev \
        ninja-build \
        pkg-config \
        python3-dev \
        python3-numpy \
        python3-pip \
        wget && \
    # Install spot
    wget -O /etc/apt/trusted.gpg.d/lrde.gpg https://www.lrde.epita.fr/repo/debian.gpg && \
    echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y libspot-dev

COPY --from=builder /usr /usr

WORKDIR /benchmark
# Python package install
RUN pip3 install --upgrade pip && pip3 install numpy scipy matplotlib

# Install itmobotics-sim
RUN pip3 install pip install git+https://github.com/ITMORobotics/itmobotics_sim.git@feature/add-collision-checking
COPY ./requirements.txt /
RUN pip3 install -r /requirements.txt

# RUN useradd -ms /bin/bash ompl
# USER ompl
# WORKDIR /home/ompl