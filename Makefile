NVCC = nvcc
NVCCFLAGS = -O3 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_75,code=sm_75 

MATIO_BASE = $(HOME)/local

MATIO_INCLUDES = -I$(MATIO_BASE)/matio/include -I$(MATIO_BASE)/hdf5/include -I$(MATIO_BASE)/zlib/include
MATIO_LIBS = -L$(MATIO_BASE)/matio/lib -L$(MATIO_BASE)/hdf5/lib -L$(MATIO_BASE)/zlib/lib -lhdf5 -lz -lmatio

EXECUTABLE_1 = executable_1
EXECUTABLE_2 = executable_2

all: $(EXECUTABLE_1) ${EXECUTABLE_2}

$(EXECUTABLE_1): main_gpu_1.cu coloringCC_gpu_1.cu
	$(NVCC) $(NVCCFLAGS) $(MATIO_INCLUDES) -o $@ $^ $(MATIO_LIBS)

$(EXECUTABLE_2): main_gpu_2.cu coloringCC_gpu_2.cu
	$(NVCC) $(NVCCFLAGS) $(MATIO_INCLUDES) -o $@ $^ $(MATIO_LIBS)


clean: 
	rm -f $(EXECUTABLE_1)
	rm -f $(EXECUTABLE_2)