FROM jupyter/pyspark-notebook:spark-3.5.0

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYSPARK_PYTHON=python3

RUN mkdir -p /home/jovyan/work

EXPOSE 8888 4040

WORKDIR /home/jovyan/work
