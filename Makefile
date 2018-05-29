all: setup run

setup:
	docker build . -t fill-in-my-blank

run:
	docker run --volume "$$PWD":/code -ti fill-in-my-blank

shell:
	docker run --volume "$$PWD":/code -ti fill-in-my-blank /bin/bash

