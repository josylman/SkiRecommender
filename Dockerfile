FROM continuumio/miniconda3

ADD ./app/requirements.txt /tmp/requirements.txt
ADD ./app/conda-requirements.txt /tmp/conda-requirements.txt

RUN pip install -qr /tmp/requirements.txt

ADD ./app /opt/app/
WORKDIR /opt/app

RUN conda install --yes --file /tmp/conda-requirements.txt
RUN pip install gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT wsgi
