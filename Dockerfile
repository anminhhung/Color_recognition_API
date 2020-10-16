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
    apt-get install -y libxrender-dev

COPY . /Vehicle_detection

RUN cd Vehicle_detection && \
    pip3 install -r requirements.txt

WORKDIR /Vehicle_detection

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    python3 -m pip install -e detectron2

EXPOSE 5003

CMD ["python3", "vehicle_detection_api.py"]