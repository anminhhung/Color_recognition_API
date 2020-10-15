FROM python:3.6-slim-stretch

RUN apt update  

RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY . /Color_recognition

RUN cd Color_recognition && \
    pip3 install -r requirements.txt \
    mkdir models \ 
    gdown --id 105A8cHr_TEz1gTf5MqiAsu3dC_8YG_45

WORKDIR /Color_recognition

EXPOSE 5001

CMD ["python3", "color_api.py""]