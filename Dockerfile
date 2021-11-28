FROM python:3.8

RUN mkdir -p /root/.kaggle
COPY kaggle.json /root/.kaggle

RUN mkdir -p /usr/src/app
COPY requirements.txt /usr/src/app
WORKDIR /usr/src/app

RUN pip install -r requirements.txt

RUN mkdir -p /usr/src/dataset
WORKDIR /usr/src/dataset
RUN kaggle competitions download -c machine-learningntut-2021-autumn-regression
RUN unzip machine-learningntut-2021-autumn-regression.zip

WORKDIR /usr/src/app
