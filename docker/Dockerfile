FROM python:3.6.6

WORKDIR /src

ADD docker/install_container_dependencies.sh /tmp/
RUN /tmp/install_container_dependencies.sh
RUN rm /tmp/install_container_dependencies.sh

# Copy requirements.txt separately so we can reuse the layer if the source
# code changes but the dependencies are the same.
COPY requirements_doc.txt requirements_dev.txt requirements.txt /tmp/
RUN pip install --upgrade -r /tmp/requirements_dev.txt
RUN rm -rf /root/.cache/pip
