.PHONY: train run test

train:
	python3 train.py

run:
	python3 run.py

test:
	python3 train.py
	python3 run.py

