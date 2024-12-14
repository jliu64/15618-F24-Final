# Target executable name
EXECUTABLE := flightroute

# CUDA compiler
NVCC := nvcc

# CUDA architecture and compiler flags
NVCCFLAGS := -O3 -m64 --gpu-architecture=compute_61 -lineinfo

# Linker flags for CUDA runtime library
LDFLAGS := -L/usr/local/cuda-11.7/lib64/ -lcudart

# Source file
CU_FILES := flightroute.cu

.PHONY: all clean

# Default target
all: $(EXECUTABLE)

# Create the output executable directly from the source file
$(EXECUTABLE): $(CU_FILES)
	$(NVCC) $(NVCCFLAGS) $(CU_FILES) $(LDFLAGS) -o $(EXECUTABLE)

# Clean up generated files
clean:
	rm -rf $(EXECUTABLE)
