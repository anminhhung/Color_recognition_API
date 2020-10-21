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

COPY . /Vehicle_detection

RUN cd Vehicle_detection && \
    python3 -m pip install -U pip &&\
    # fix bug can not install skbuild
    python3 -m pip install -U setuptools &&\
    pip3 install -r requirements.txt &&\
    mkdir models &&\ 
    cd models &&\
    # weight + config file
    gdown --id 1RAzksiPKelOk6PxcEVWMsvOxfySecD9i &&\
    gdown --id 1himth8An9mn2YzGGdsve9WLXegMR3NxK &&\
    # file classes
    gdown --id 1SDaqVnbJgrWms8I9lmuFlwupqIsb0DQR

# cd models &&\
# gdown --id 1vwXfXkvMwfwZfTO7k8vE4GteYzuyy7jW &&\
# gdown --id 1SDaqVnbJgrWms8I9lmuFlwupqIsb0DQR

WORKDIR /Vehicle_detection

# RUN git clone https://github.com/facebookresearch/detectron2.git && \
#     python3 -m pip install -e detectron2

EXPOSE 5003

CMD ["python3", "vehicles_detection_api.py"]