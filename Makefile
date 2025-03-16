# Use bash for all commands
SHELL := /bin/bash

# Path to Conda base directory
CONDA_BASE := $(shell conda info --base)

# Env args
ENV_NAME=triton-flash-env

build:
	. $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda create --name $(ENV_NAME) python=3.12 -y && \
	conda activate $(ENV_NAME) && \
	pip install -r requirements.txt

test:
	. $(CONDA_BASE)/etc/profile.d/conda.sh  &&\
	conda activate $(ENV_NAME) &&\
	pytest

format:
	. $(CONDA_BASE)/etc/profile.d/conda.sh  &&\
	conda activate $(ENV_NAME) &&\
	ruff format
	
check:
	. $(CONDA_BASE)/etc/profile.d/conda.sh  &&\
	conda activate $(ENV_NAME) &&\
	ruff check

fix:
	. $(CONDA_BASE)/etc/profile.d/conda.sh  &&\
	conda activate $(ENV_NAME) &&\
	ruff check --fix

clean:
	conda remove --name $(ENV_NAME) --all -y