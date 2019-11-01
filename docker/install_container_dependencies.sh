apt-get update

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

apt-get install -y --no-install-recommends \
  python-opencv \
  libsuitesparse-dev \
  libboost-dev \
  libspatialindex-dev \
  git-lfs
rm -rf /var/lib/apt/lists/*

# numpy is an install dependency for blmath and lace, so we install it with
# the container dependencies.
pip install numpy==1.13.1
