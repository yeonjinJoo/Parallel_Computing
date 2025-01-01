# Compiler
NVCC = nvcc
CUDAFLAGS = -O3

# Target
all: km_cuda

# Build km_cuda
km_cuda: km_cuda.cu
	$(NVCC) $(CUDAFLAGS) -o km_cuda km_cuda.cu

# Clean up
clean:
	rm -f km_cuda
