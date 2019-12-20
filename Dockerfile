FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter
# see: https://www.tensorflow.org/install/docker
# see: https://hub.docker.com/r/tensorflow/tensorflow/

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      git \
      wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /work/

WORKDIR /work

RUN pip install -r requirements.txt

ENV PYTHONPATH='/work/:$PYTHONPATH'

CMD ["bash"]
