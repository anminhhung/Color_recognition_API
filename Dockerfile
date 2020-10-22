FROM ubuntu:18.04 

RUN apt-get update

RUN apt-get install -y git \
    software-properties-common \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install python3.6 -y && \
    apt install python3-distutils -y && \
    apt install python3.6-dev -y && \
    apt install build-essential -y && \
    apt-get install python3-pip -y && \
    apt update && apt install -y libsm6 libxext6 && \
    apt-get install -y libxrender-dev && \ 
    apt install libgl1-mesa-glx -y
    #ImportError: libGL.so.1: cannot open shared object file: No such file or directory

COPY . /vehicles_tracking

RUN cd vehicles_tracking && \
    python3 -m pip install -U pip &&\
    # fix bug can not install skbuild
    python3 -m pip install -U setuptools &&\
    pip3 install -r requirements.txt

WORKDIR /vehicles_tracking

EXPOSE 5004

CMD ["python3", "vehicles_tracking_api.py"]