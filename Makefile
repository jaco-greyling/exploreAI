install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black --line-length=120 predictive-bets/*.py &&\
		isort predictive-bets/

lint: 
	# TODO: check some linting

test: 
	# TODO: Add tests

all: install format