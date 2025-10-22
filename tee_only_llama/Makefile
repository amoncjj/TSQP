# SPDX-License-Identifier: LGPL-3.0-or-later

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

PYTHON_BIN ?= $(realpath $(shell sh -c "command -v python3"))
PROJ_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

TEE_ONLY_MANIFEST := tee_only.manifest
TEE_ONLY_SGX := $(TEE_ONLY_MANIFEST).sgx
TEE_ONLY_SIG := tee_only.sig
TEE_ONLY_APP := tee_runner.py

.PHONY: all
all: $(TEE_ONLY_MANIFEST)
ifeq ($(SGX),1)
all: $(TEE_ONLY_SGX) $(TEE_ONLY_SIG)
endif

$(TEE_ONLY_MANIFEST): tee_only.manifest.template $(TEE_ONLY_APP)
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dentrypoint=$(PYTHON_BIN) \
		$< > $@

$(TEE_ONLY_MANIFEST).sgx $(TEE_ONLY_SIG): $(TEE_ONLY_MANIFEST)
	gramine-sgx-sign --manifest $< --output $<.sgx

.PHONY: clean
clean:
	$(RM) *.token *.sig *.manifest.sgx *.manifest

.PHONY: distclean
distclean: clean
	$(RM) tee_only_results.json
