all: setup run

setup:
	docker build . -t fill-in-my-blank

run:
	docker run --volume "$$PWD":/code -ti fill-in-my-blank

test-data:
	docker run --volume "$$PWD":/code -ti fill-in-my-blank python data/to_npy.py

shell:
	docker run --volume "$$PWD":/code -ti fill-in-my-blank /bin/bash

