all: install-dev activate-environment generate-npy run-test

setup:
	docker build . -t fill-in-my-blank

run:
	docker run --volume "$$PWD":/code -ti fill-in-my-blank python src/test.py 

shell:
	docker run --volume "$$PWD":/code -ti fill-in-my-blank /bin/bash

install-dev:
	brew install pipenv | true
	pipenv install --dev
	#TODO : check if it's there before setting again
	echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc

activate-environment:
	pipenv shell | true

generate-npy: activate-environment
	cd ./data ; python to_npy.py
	
run-test:
	cd ./src/test ; python test.py

