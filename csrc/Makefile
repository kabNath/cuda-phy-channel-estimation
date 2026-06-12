# Build the MMSE-apply kernel-optimization case study.
#   make                  # build for an RTX 40-series (sm_89)
#   make ARCH=sm_75       # Turing (T4)         ARCH=sm_80 (A100)  ARCH=sm_90 (H100)
#   make selftest         # build + correctness check
#   make bench            # build + benchmark the three variants
#   make compile-check    # compile only, no GPU needed (used by CI)
NVCC      ?= nvcc
ARCH      ?= sm_89
NVCCFLAGS  = -O3 -std=c++14 -arch=$(ARCH) -lineinfo
LIBS       = -lcublas

mmse_apply: mmse_apply.cu
	$(NVCC) $(NVCCFLAGS) mmse_apply.cu -o mmse_apply $(LIBS)

.PHONY: selftest bench compile-check clean
selftest: mmse_apply
	./mmse_apply --selftest

bench: mmse_apply
	./mmse_apply --bench

# Compiles the translation unit without linking or a GPU — safe on a CI runner.
compile-check:
	$(NVCC) $(NVCCFLAGS) -c mmse_apply.cu -o /tmp/mmse_apply.o

clean:
	rm -f mmse_apply *.o
