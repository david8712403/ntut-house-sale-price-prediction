FROM python:3.8

# 複製kaggle API key檔案
RUN mkdir -p /root/.kaggle
COPY kaggle.json /root/.kaggle

# 安裝python相依套件
RUN mkdir -p /usr/src/app
COPY requirements.txt /usr/src/app
WORKDIR /usr/src/app
RUN pip install -r requirements.txt

# 從kaggle下載房價預測dataset並解壓縮
RUN mkdir -p /usr/src/dataset
WORKDIR /usr/src/dataset
RUN kaggle competitions download -c machine-learningntut-2021-autumn-regression
RUN unzip machine-learningntut-2021-autumn-regression.zip

WORKDIR /usr/src/app
