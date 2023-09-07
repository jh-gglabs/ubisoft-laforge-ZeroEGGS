FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/local/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3.8 \
        python3.8-venv \
        python3-pip \
        unzip \
        sox

SHELL ["/bin/bash", "-c"]

COPY requirements.txt .
RUN python3.8 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade setuptools && \
    pip install -U pip && \
    pip install -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
