# Makefile for Fraud Detection Pipeline Docker Image

# Default architecture
ARCH ?= amd64

# Docker image name and tag
IMAGE_NAME := fraud-detection-pipeline
IMAGE_TAG := latest

# Docker build command
DOCKER_BUILD := docker build -t $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: build-amd64 build-arm64 build

# Build for AMD64 architecture
build-amd64:
	$(DOCKER_BUILD) --platform linux/amd64 .

# Build for ARM64 architecture
build-arm64:
	$(DOCKER_BUILD) --platform linux/arm64 .

# Build for the specified architecture (default: amd64)
build:
ifeq ($(ARCH),amd64)
	@$(MAKE) build-amd64
else ifeq ($(ARCH),arm64)
	@$(MAKE) build-arm64
else
	@echo "Unsupported architecture: $(ARCH). Use 'amd64' or 'arm64'."
	@exit 1
endif

# Help target
help:
	@echo "Available targets:"
	@echo "  build-amd64  : Build Docker image for AMD64 architecture"
	@echo "  build-arm64  : Build Docker image for ARM64 architecture"
	@echo "  build        : Build Docker image for the specified architecture (default: amd64)"
	@echo "                 Use ARCH=arm64 to build for ARM64"
	@echo "  help         : Show this help message"

# Default target
.DEFAULT_GOAL := help