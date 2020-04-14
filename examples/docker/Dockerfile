# FROM python:3.7
FROM tensorflow/tensorflow:2.2.0rc2-py3

RUN apt-get update && apt-get -y install git

COPY requirements.txt /tmp

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY server-requirements.txt  /tmp

RUN pip install --no-cache-dir -r /tmp/server-requirements.txt

WORKDIR /app

COPY src/* ./

CMD ["uwsgi", "--ini", "wsgi.ini"]
