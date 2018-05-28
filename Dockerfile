FROM python:3.6.5-jessie

RUN apt-get update
RUN apt-get install -y cmake

ADD requirements.txt ./
RUN pip3 install -r requirements.txt

RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

WORKDIR /code
COPY . /code
