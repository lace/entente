FROM python:2.7.15

WORKDIR /src

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
  python-opencv \
  libsuitesparse-dev \
  libboost-dev \
  libcgal-dev \
  swig
RUN rm -rf /var/lib/apt/lists/*

# numpy is an install dependency for blmath and lace.
RUN pip install numpy==1.13.1

# Copy requirements.txt separately so we can re-use the layer if the source
# code changes but the dependencies are the same.
COPY requirements_dev.txt requirements.txt /tmp/
RUN pip install --upgrade -r /tmp/requirements_dev.txt

RUN rm -rf /root/.cache/pip
