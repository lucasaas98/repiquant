SHELL=/bin/bash

# dev
deps:
	echo "Installing dependencies"
	python3 -m venv .venv && source .venv/bin/activate && pip3 install --upgrade -r requirements.txt

run:
	echo "Starting the server!"
	source .venv/bin/activate && python3 -m flask run --host=0.0.0.0

format:
	echo "Running black, isort and flake8"
	source .venv/bin/activate && black . && isort . && flake8 .