# Makefile for Image Similarity System

# Variables
CC = g++
CFLAGS = -O3 -mcpu=cortex-a72 -mfpu=neon
PYTHON = python3
PYBIND11_DIR = $(shell python3 -m pybind11 --includes)
BUILD_DIR = build
SRC_DIR = core/neon_optimized
BINDINGS_DIR = core/bindings

# Targets
.PHONY: all clean test run

all: build_python

build_python: $(BUILD_DIR)/neon_ops.so

$(BUILD_DIR)/neon_ops.so: $(BINDINGS_DIR)/similarity.cpp
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -o $@ $< $(PYBIND11_DIR)

clean:
	rm -rf $(BUILD_DIR)/*.so

test:
	$(PYTHON) -m unittest discover tests/

run:
	$(PYTHON) webapp/app.py
