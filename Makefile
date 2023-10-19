# Define the name of the Docker network
NETWORK_NAME = mynetwork

# Define the name of the Docker image to run
IMAGE_NAME = showdown

# Define the name of the Docker image to build
AGENT_IMAGE_NAME = poke-agent

# Define the path to the directory to mount
MOUNT_DIR = ../pokemon_league

.PHONY: start build build-showdown build-agent run-agent

build:
	# Build the Docker image from the Dockerfile
	docker build -t $(IMAGE_NAME) -f dockerfile.showdown .
	docker build -t $(AGENT_IMAGE_NAME) -f dockerfile.agent .

build-showdown:
	docker build -t $(IMAGE_NAME) -f dockerfile.showdown .

build-agent:
	# Build the Docker image from the Dockerfile
	docker build -t $(AGENT_IMAGE_NAME) -f dockerfile.agent .

start:
	# Create the Docker network if it doesn't already exist
	docker network inspect $(NETWORK_NAME) >/dev/null 2>&1 || docker network create $(NETWORK_NAME)
	# Start the Docker container on the network
	docker run --name $(IMAGE_NAME) --network $(NETWORK_NAME) -p 8000:8000 $(IMAGE_NAME)

run-agent:
	# Start the Docker container for the agent and execute the given command
	docker run --rm -it --network $(NETWORK_NAME) -v $(abspath $(MOUNT_DIR)):/workspace/pokemon_league $(AGENT_IMAGE_NAME) $(CMD)

