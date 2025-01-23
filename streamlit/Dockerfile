FROM continuumio/miniconda3

WORKDIR /home/app

RUN apt-get update -y 

COPY . /home/app

RUN pip install -r requirements.txt

CMD streamlit run --server.port $PORT app.py