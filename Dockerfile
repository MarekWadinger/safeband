# Use an official Python runtime as a parent image
FROM python:3.11.3-slim
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
# # Copy the current directory contents into the container at /app (except .dockerignore)
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -q --no-cache-dir -U pip setuptools setuptools_rust wheel
RUN pip install -q --no-cache-dir -U -r requirements.txt

# Run rpc_client.py when the container launches
CMD ["python", "rpc_client.py", "--topic", \
     "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power", \
     "shellies/Shelly3EM-Main-Switchboard-C/emeter/1/power", \
     "shellies/Shelly3EM-Main-Switchboard-C/emeter/2/power"]
