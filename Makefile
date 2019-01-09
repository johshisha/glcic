export DOCKERFILE=docker/Dockerfile
export DOCKERFILE_GPU=docker/Dockerfile-gpu

all: setup run

setup:
	docker build . -t fill-in-my-blank -f $(DOCKERFILE)

setup-gpu:
	docker build . -t fill-in-my-blank -f $(DOCKERFILE_GPU)

run:
	docker run --rm --volume "$$PWD":/code -ti fill-in-my-blank

shell:
	docker run --rm --volume "$$PWD":/code -ti fill-in-my-blank /bin/bash
