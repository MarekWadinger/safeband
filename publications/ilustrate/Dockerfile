# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install system requirements
RUN apt-get -qq update && apt-get install -y gcc g++
RUN apt-get -qq update && apt-get install -y git curl
RUN apt-get -qq install -y --fix-missing tini texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="${PATH}:/root/.cargo/bin"

# # Set the working directory to /app
WORKDIR /app

# Install pip requirements
RUN pip install -q -U pip setuptools setuptools_rust wheel
RUN pip install jupyter -U
COPY requirements.txt /app
RUN pip install -q --no-cache-dir -U -r requirements.txt


COPY . /app

RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 8888
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
