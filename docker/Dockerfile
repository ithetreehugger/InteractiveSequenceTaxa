FROM ubuntu:18.04

MAINTAINER Diana <yeng2@jhu.edu>

RUN apt-get update && apt-get install -y python3 python3-pip git
RUN pip3 install pipenv

RUN git clone https://github.com/ithetreehugger/InteractiveSequenceTaxa.git
WORKDIR InteractiveSequenceTaxa
RUN tar -xvf lineage-data.tar.gz
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8


RUN pipenv install
EXPOSE 5006
ENTRYPOINT ["pipenv", "run", "bokeh", "serve", "--websocket-max-message-size", "104857600", "--show", "metagenomics", "--allow-websocket-origin=*"]

