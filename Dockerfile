FROM pytorch/pytorch

COPY requirements.txt .
RUN apt-get update 
RUN apt-get install -y vim tmux

# Opencv needs this apparently...
RUN apt-get install -y libglib2.0-0
RUN apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev


RUN pip install -r requirements.txt

CMD /bin/bash

