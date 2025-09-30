# Simple Makefile to build a CUDA + MPI example using UCX device-side API

# Customize these if needed
UCX_PREFIX ?= /home/snordmann/ucx_build
MPI_CXX    ?= mpicxx
NVCC       ?= nvcc

TARGET     := g2g_ucx_device
SRCS       := main.cu
OBJS       := $(SRCS:.cu=.o)

# NOTE: The following include of /home/snordmann/ucx/src/ is required for uct/ib/mlx5/gdaki/gdaki.cuh,
#       but this is not a typical usage and may indicate a non-standard include dependency.
INC_FLAGS  := -I$(UCX_PREFIX)/include -I/home/snordmann/ucx/src/ -I/opt/mellanox/doca/include
LIB_FLAGS  := -L$(UCX_PREFIX)/lib -lucp -luct -lucs -lcudart

# Add rpath so the runtime linker can find UCX shared libs
RPATH_FLAGS := -Wl,-rpath,$(UCX_PREFIX)/lib

CXXFLAGS   ?= -O2 -g
NVCCFLAGS  ?= -O2 -g -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr
# Use MPI wrapper as host compiler so mpi.h is found during NVCC host compilation
NVCCFLAGS  += -ccbin $(MPI_CXX)
## Build only for NVIDIA H100 (Hopper, SM90)
NVCCFLAGS  += -gencode arch=compute_90,code=sm_90

all: $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INC_FLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(MPI_CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS) $(RPATH_FLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)




