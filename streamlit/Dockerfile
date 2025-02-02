FROM continuumio/miniconda3

RUN apt-get update -y 

WORKDIR /home/app

COPY . /home/app

RUN mkdir -p data
RUN chmod 777 data

RUN pip install -r requirements.txt

CMD streamlit run --server.port $PORT streamlit_app.py