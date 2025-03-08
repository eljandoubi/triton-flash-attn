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

check:
	. $(CONDA_BASE)/etc/profile.d/conda.sh  &&\
	conda activate $(ENV_NAME) &&\
	pytest

lint:
	. $(CONDA_BASE)/etc/profile.d/conda.sh  &&\
	conda activate $(ENV_NAME) &&\
	pylint */*.py

autpep:
	. $(CONDA_BASE)/etc/profile.d/conda.sh  &&\
	conda activate $(ENV_NAME) &&\
	autopep8 --in-place --recursive --aggressive --aggressive .
	
format: autpep lint

clean:
	conda remove --name $(ENV_NAME) --all -y