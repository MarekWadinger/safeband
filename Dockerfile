# Use an official Python runtime as a parent image
FROM python:3.11-slim
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install system requirements
#  gcc for C compilation
RUN apt-get -qq update && apt-get install -y gcc g++
RUN apt-get -qq update && apt-get install -y git curl

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="${PATH}:/root/.cargo/bin"

# # Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install -q -U pip setuptools setuptools_rust wheel
COPY requirements.txt /app
RUN pip install -q --no-cache-dir -U -r requirements.txt

# # Copy the current directory contents into the container at /app (except .dockerignore)
COPY . /app

WORKDIR /app/examples

# Run rpc_client.py when the container launches
CMD ["python", "comparison_diagnostics.py"]
