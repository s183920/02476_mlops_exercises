# Base image
FROM python:3.11-slim
# FROM  nvcr.io/nvidia/pytorch:22.07-py3

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_exercises/ mlops_exercises/
COPY data/ data/
COPY Makefile Makefile

# install dependencies
WORKDIR /
RUN make requirements_docker

# run training
ENTRYPOINT ["python", "-u", "mlops_exercises/train_model.py", "train", "--epochs=3"]
# ENTRYPOINT ["python", "-u", "-m", "pip", "list"]
