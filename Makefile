NVCC = nvcc
NVCCFLAGS = -O3 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_75,code=sm_75 

MATIO_BASE = $(HOME)/local

MATIO_INCLUDES = -I$(MATIO_BASE)/matio/include -I$(MATIO_BASE)/hdf5/include -I$(MATIO_BASE)/zlib/include
MATIO_LIBS = -L$(MATIO_BASE)/matio/lib -L$(MATIO_BASE)/hdf5/lib -L$(MATIO_BASE)/zlib/lib -lhdf5 -lz -lmatio

EXECUTABLE_TPN = executable_TPN
EXECUTABLE_WPN = executable_WPN

all: $(EXECUTABLE_TPN) ${EXECUTABLE_WPN}

$(EXECUTABLE_TPN): main_gpu_TPN.cu coloringCC_gpu_TPN.cu
	$(NVCC) $(NVCCFLAGS) $(MATIO_INCLUDES) -o $@ $^ $(MATIO_LIBS)

$(EXECUTABLE_WPN): main_gpu_WPN.cu coloringCC_gpu_WPN.cu
	$(NVCC) $(NVCCFLAGS) $(MATIO_INCLUDES) -o $@ $^ $(MATIO_LIBS)


clean: 
	rm -f $(EXECUTABLE_TPN)
	rm -f $(EXECUTABLE_WPN)