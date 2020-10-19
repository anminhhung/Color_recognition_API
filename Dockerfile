FROM python:3.6-slim-stretch

RUN apt update  

RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY . /vehicle_service

RUN cd vehicle_service && \
    pip3 install -r requirements.txt

WORKDIR /vehicle_service

EXPOSE 5050

CMD ["python3", "vehicles_service_api.py"]
