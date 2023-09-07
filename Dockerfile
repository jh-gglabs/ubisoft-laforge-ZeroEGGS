FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/local/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3.8 \
        python3.8-venv

SHELL ["/bin/bash", "-c"]

COPY requirements.txt .
RUN python3.8 -m venv venv && \
    source venv/bin/activate && \
    pip install -r requirements.txt

COPY . .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
