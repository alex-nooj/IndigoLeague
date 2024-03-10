# Start with a smaller base image if possible, or stick with the necessary CUDA base
FROM nvidia/cuda:12.3.1-base-ubuntu20.04 as builder

# Set arguments and environment variables
ARG USER_ID=1000
ARG GROUP_ID=1000
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies, clean up in the same layer to reduce size
RUN apt-get update && apt-get install -y \
    curl \
    g++ \
    python3.8 \
    python3-pip \
    libcusparse-11-2 && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Prepare the workspace directory
ARG WORKSPACE_DIR=/workspaces/IndigoLeague
WORKDIR ${WORKSPACE_DIR}

# Start the final stage
FROM nvidia/cuda:12.3.1-base-ubuntu20.04 as prod

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV DEBIAN_FRONTEND=noninteractive

# Install Python in the final image
RUN apt-get update && apt-get install -y python3.8 python3-pip && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

ARG WORKSPACE_DIR=/workspaces/IndigoLeague

# Copy only the necessary files for building the Python environment
COPY requirements.txt setup.py ${WORKSPACE_DIR}/

# Recreate the user to avoid running as root
RUN addgroup --gid $GROUP_ID usergroup && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN pip install -r ${WORKSPACE_DIR}/requirements.txt && \
    pip install ${WORKSPACE_DIR}/.

USER user

WORKDIR ${WORKSPACE_DIR}

# Copy only the artifacts needed at runtime from the builder stage
COPY --from=builder ${WORKSPACE_DIR} ${WORKSPACE_DIR}
COPY --chown=user:user . ${WORKSPACE_DIR}/

# Set the default command
CMD ["python", "-m", "indigo_league.main"]
