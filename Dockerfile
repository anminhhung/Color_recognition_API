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
    gdown --id 11VJ5gsAdYu0LtCWKZYKMz76BRzPMChDr &&\
    gdown --id 1DP2kHKfY7Ua8qB4HU-t8lFnI4DnolIYH &&\
    gdown --id 1o7EYuJYwbMdGacRTtSSOOUppWqZH3-Y8

WORKDIR /vehicles_recognition

EXPOSE 5001

CMD ["python3", "vehicles_recog_api.py"]