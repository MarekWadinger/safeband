# --- Builder Stage ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Install the project into `/app`
WORKDIR /app

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

# Install system requirements
#  gcc for C compilation
RUN apt-get -qq update && apt-get install -y gcc g++
#  git for version control
RUN apt-get -qq update && apt-get install -y git curl


# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="${PATH}:/root/.cargo/bin"

# Install the project's dependencies using the lockfile and settings
# TODO: Add comments on mount types and why we use them
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable --group example


# Copy the project into the intermediate image
COPY . /app

# --- Final Stage ---
# This stage builds the final, lean image
FROM python:3.12-slim

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app /app

# Make sure the environment is in the PATH
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app/examples

# Run rpc_client.py when the container launches
CMD ["python", "comparison_diagnostics.py"]
