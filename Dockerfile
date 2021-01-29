FROM python:3.6-slim-stretch

RUN apt update  

RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY . /vehicles_recognition

RUN cd vehicles_recognition && \
    pip3 install -r requirements.txt &&\
    mkdir models &&\ 
    cd models &&\
    gdown --id 1CUgExRR225igNqrlmIDV9fHy08oD87mm &&\
    gdown --id 1-0JFpymC1eyMPmfA4DFfwhtW7lx9Q31Q &&\
    gdown --id 11GQFH5UFf-assuEshRWHphwxNgJ-uZqD

WORKDIR /vehicles_recognition

EXPOSE 5001

CMD ["python3", "vehicles_recog_api.py"]
