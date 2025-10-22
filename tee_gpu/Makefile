# SPDX-License-Identifier: LGPL-3.0-or-later

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

PYTHON_BIN ?= $(realpath $(shell sh -c "command -v python3"))
PROJ_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

GRPC_STUBS := msg_pb2.py msg_pb2_grpc.py
GRPC_PROTO := msg.proto

TEE_GPU_MANIFEST := tee_runner.manifest
TEE_GPU_SGX := $(TEE_GPU_MANIFEST).sgx
TEE_GPU_SIG := tee_runner.sig
TEE_GPU_APP := tee_runner.py

SERVER_MANIFEST := server.manifest
SERVER_MANIFEST_SGX := $(SERVER_MANIFEST).sgx
SERVER_SIG := server.sig
SERVER_APP := server.py

.PHONY: all
all: $(TEE_GPU_MANIFEST)
ifeq ($(SGX),1)
all: $(TEE_GPU_SGX) $(TEE_GPU_SIG) $(SERVER_MANIFEST_SGX) $(SERVER_SIG)
endif

$(GRPC_STUBS): $(GRPC_PROTO)
	$(PYTHON_BIN) -m grpc_tools.protoc \
		-I $(PROJ_ROOT) \
		--python_out=$(PROJ_ROOT) \
		--grpc_python_out=$(PROJ_ROOT) \
		$(PROJ_ROOT)/$<

$(TEE_GPU_MANIFEST): tee_runner.manifest.template $(TEE_GPU_APP) $(GRPC_STUBS)
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dentrypoint=$(PYTHON_BIN) \
		$< > $@

$(TEE_GPU_MANIFEST).sgx $(TEE_GPU_SIG): $(TEE_GPU_MANIFEST)
	gramine-sgx-sign --manifest $< --output $<.sgx

$(SERVER_MANIFEST): pytorch.manifest.template $(SERVER_APP) $(GRPC_STUBS)
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dentrypoint=$(PYTHON_BIN) \
		$< > $@

$(SERVER_MANIFEST).sgx $(SERVER_SIG): $(SERVER_MANIFEST)
	gramine-sgx-sign --manifest $< --output $<.sgx

.PHONY: clean
clean:
	$(RM) *.token *.sig *.manifest.sgx *.manifest $(GRPC_STUBS)

.PHONY: distclean
distclean: clean
	$(RM) tee_gpu_benchmark.json tee_only_result.json
