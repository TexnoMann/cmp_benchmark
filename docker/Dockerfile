FROM nvidia/opengl:1.2-glvnd-devel-ubuntu20.04
SHELL ["/bin/bash", "-ci"]

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

# OMPL with bindings installation
RUN apt update && apt install -y git cmake clang python3 python3-dev python3-pip python3-distutils libpython3-dev \
    libeigen3-dev curl gnupg lsb-release wget software-properties-common minizip ffmpeg vim nano
# OMPL with bindings installation
RUN mkdir /ompl
COPY ./docker/3rd_party/install-ompl-ubuntu.sh /ompl
RUN apt update && apt install -y git cmake clang python3 python3-dev python3-pip python3-distutils libpython3-dev \
    libeigen3-dev curl gnupg lsb-release wget software-properties-common minizip ffmpeg vim nano libboost-dev \
    g++ cmake pkg-config libboost-serialization-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libboost-test-dev libeigen3-dev libode-dev wget libyaml-cpp-dev \
    libboost-python-dev && \
    pip${PYTHONV} install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus && \
    cd /ompl && ./install-ompl-ubuntu.sh --python && ./install-ompl-ubuntu.sh --app && \
    cd /ompl && git clone --recurse-submodules https://github.com/ompl/omplapp.git  && cd omplapp \
    && mkdir -p build/Release && cd build/Release && cmake ../.. && make -j 4 update_bindings && make -j 4

# Python package install
RUN pip3 install --upgrade pip && pip3 install numpy scipy matplotlib

WORKDIR /benchmark

# Install itmobotics-sim
RUN pip3 install pip install git+https://github.com/ITMORobotics/itmobotics_sim.git@feature/add-collision-checking
COPY ./requirements.txt /
RUN pip3 install -r /requirements.txt