# Define the name of the Docker network
NETWORK_NAME = host

# Define the name of the Docker image to run
IMAGE_NAME = showdown

# Version name
VERSION=$(shell cat VERSION)

# Define the name of the Docker image to build
AGENT_IMAGE_NAME = poke-agent

# Define the path to the directory to mount
MOUNT_DIR = ../pokemon_league

UID=$(shell id -u)
GID=$(shell id -g)

.PHONY: start build build-showdown build-agent run-agent

build:
	# Build the Docker image from the Dockerfile
	docker build -t $(IMAGE_NAME) -f dockerfile.showdown .
	docker build -t $(AGENT_IMAGE_NAME):$(VERSION) -f dockerfile.agent .

build-showdown:
	docker build -t $(IMAGE_NAME) -f dockerfile.showdown .

build-agent:
	# Build the Docker image from the Dockerfile
	DOCKER_BUILDKIT=1 docker build --build-arg USER_ID=$(UID) --build-arg GROUP_ID=$(GID) -t $(AGENT_IMAGE_NAME):$(VERSION) -f dockerfile.agent .

start:
	docker run --name $(IMAGE_NAME) --network $(NETWORK_NAME) -p 8000:8000 $(IMAGE_NAME)

run-agent:
	# Start the Docker container for the agent and execute the given command
	docker run --rm --network $(NETWORK_NAME) -m 3g --gpus=all -v $(abspath $(MOUNT_DIR)):/workspaces/pokemon_league $(AGENT_IMAGE_NAME):$(VERSION)
