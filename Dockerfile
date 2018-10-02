# This Dockerfile is used to create a static build on Zeit Now for previewing
# documentation in pull requests.
# https://zeit.co/docs/static-deployments/builds/building-with-now
#
# The underlying docker image is defined in docker/entente-ci-py2.7/Dockerfile.

FROM laceproject/entente-ci-py2.7:0.1.0
WORKDIR /src
COPY . /src
ENV PYTHONPATH /src

RUN ./dev.py doc
RUN cp -r doc/build /public
