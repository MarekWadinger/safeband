# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install system requirements
RUN apt-get -qq update && apt-get install -y tini gcc
#  g++ curl
RUN apt -qq install -y --fix-missing texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super 

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install pip requirements
# Install from local wheel
# COPY *.whl .
#RUN pip install --no-index --no-deps *.whl
RUN pip install -q --no-cache-dir -U pip
RUN pip install jupyter -U
COPY requirements.txt .
RUN python -m pip install -r requirements.txt


WORKDIR /app
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
