# Makefile for TSQP TEE+GPU Project
# Simplified for ZeroMQ-based communication

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)
PYTHON_BIN ?= python3

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

PROJ_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

TEE_GPU_MANIFEST := tee_runner.manifest
TEE_GPU_APP := tee_runner.py

SERVER_MANIFEST := pytorch.manifest
SERVER_APP := server.py

.PHONY: all
all: $(TEE_GPU_MANIFEST) $(SERVER_MANIFEST)

ifeq ($(SGX),1)
all: $(TEE_GPU_MANIFEST).sgx $(TEE_GPU_MANIFEST).sig $(SERVER_MANIFEST).sgx $(SERVER_MANIFEST).sig
endif

$(TEE_GPU_MANIFEST): tee_runner.manifest.template $(TEE_GPU_APP)
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Dentrypoint=$(PYTHON_BIN) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		$< > $@

%.manifest.sgx %.sig: %.manifest
	gramine-sgx-sign --manifest $< --output $<.sgx

$(SERVER_MANIFEST): pytorch.manifest.template $(SERVER_APP)
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Dentrypoint=$(PYTHON_BIN) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		$< > $@

.PHONY: clean
clean:
	$(RM) *.token *.sig *.manifest.sgx *.manifest

.PHONY: distclean
distclean: clean
	$(RM) -r __pycache__ *.pyc
