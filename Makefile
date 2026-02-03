PROJECT_NAME = DSLR
PYTHON_VERSION = 3.13.0
PYTHON_INTERPRETER = python

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

test:
	python -m unittest discover -s tests


.PHONY: clean test
