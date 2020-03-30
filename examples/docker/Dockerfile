FROM python:3.7

COPY requirements.txt /tmp

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY server-requirements.txt  /tmp

RUN pip install --no-cache-dir -r /tmp/server-requirements.txt

WORKDIR /app

COPY src/* ./

CMD ["uwsgi", "--ini", "wsgi.ini"]
