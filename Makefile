all: install-dev activate-environment generate-npy run-test

install-dev:
	brew install pipenv | true
	pipenv install --dev
	#TODO : check if it's there before setting again
	echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc

activate-environment:
	pipenv shell | true

generate-npy: activate-environment
	cd ./data ; python to_npy.py
	
run-test: activate-environment
	cd ./src/test ; python test.py

