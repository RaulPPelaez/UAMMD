
CUDA_VERSION=8.0

CXX=g++
CC=$(CXX)
NVCC=/usr/local/cuda-$(CUDA_VERSION)/bin/nvcc

#EXPERIMENTAL=1  #Uncomment to compile with experimental modules

###############################Put here your system configuration:#####################################

#the code will look here for the cuda and thrust sources
CUDA_INCLUDE=/usr/local/cuda-$(CUDA_VERSION)/include
#The place where the cuda libs are, libcuda.so, libcudart, libcurand..
CUDA_LIBS=/usr/local/cuda-$(CUDA_VERSION)/lib64
#The code will look for a folder named cub here, you can create a symlink if you have a folder i.e. cub-1.5.2
CUB_INCLUDE=/usr/local/include

ARCH ?= 52 #35

#You can place your custom files either here or below in the diferent OBJS_*/CUOBJS_* variables. Dont forget the folders if you put it here!
#These will be compiled with $(CXX)
CUSTOM_OBJS= 

#These ones with $(NVCC)
CUSTOM_CUOBJS= Interactor/NeighbourList/Sorter.o Interactor/NeighbourList/CellList.o Interactor/NeighbourList/NeighbourList.o

FORTRAN_OBJS =


INPUT_FILE= Driver/SimulationConfig.cpp

INPUT_FILE_OBJ= $(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(INPUT_FILE)))

#############################################################################################################################

#DEBUG
OPTIMCPU=-O3 -funroll-loops -m64 -ffinite-math-only -fno-signaling-nans -fno-math-errno -fno-signed-zeros -frename-registers -march=native -fPIC

ifdef DEBUG
OPTIMCPU= -g -O0 -fPIC
endif


INCLUDEDIRS= . Integrator Interactor Interactor/NeighbourList Driver utils globals Measurable third_party misc $(CUDA_INCLUDE) $(CUDA_INCLUDE) $(CUB_INCLUDE)

INCLUDEFLAGS= $(addprefix -I, $(INCLUDEDIRS))



CFLAGS= -pthread $(OPTIMCPU)   $(INCLUDEFLAGS)

ifdef EXPERIMENTAL
CFLAGS += -DEXPERIMENTAL
endif


ifdef EXPERIMENTAL
CFLAGS += -DEXPERIMENTAL
endif


CXXFLAGS=$(CFLAGS) -std=c++11 #-flto 


$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ifdef DEBUG
OPTIMGPU= -G -O0 -g
else
OPTIMGPU= -O3
endif
NVFLAGS= $(OPTIMGPU) -std=c++11  -ccbin=$(CXX)   --compiler-options='$(CFLAGS)' -lineinfo -m64 $(GENCODE_FLAGS) -src-in-ptx  $(INCLUDEFLAGS) -Xptxas -dlcm=cg #-Xptxas -warn-spills

LDFLAGS= -L$(CUDA_LIBS)
LDLIBS= -lcuda -lcudart -lpthread -lcurand -lcusolver -lcublas -lgfortran -lgomp -lnvToolsExt -lcufft -llapacke


OBJS_INTERACTOR_EXPERIMENTAL= PairForcesAlt.o
CUOBJS_INTERACTOR_EXPERIMENTAL= PairForcesAltGPU.o


THIRD_PARTY_OBJS= bravais.o

OBJS_INTERACTOR= BondedForces.o

OBJS_BDHI=    #BrownianHydrodynamicsEulerMaruyama.o DiffusionBDHI.o BrownianNoiseBDHI.o
CUOBJS_BDHI= BrownianHydrodynamicsEulerMaruyama.o   BDHI_Cholesky.o BDHI_Lanczos.o BDHI_PSE.o

OBJS_INTEGRATOR= VerletNVT.o VerletNVE.o BrownianEulerMaruyama.o  $(addprefix BDHI/, $(OBJS_BDHI))
OBJS_MEASURABLE= Measurable.o EnergyMeasure.o


CUOBJS_INTERACTOR= PairForces.o PairForcesDPD.o BondedForcesGPU.o #NBodyForcesGPU.o
CUOBJS_INTEGRATOR= VerletNVTGPU.o VerletNVEGPU.o BrownianEulerMaruyamaGPU.o $(addprefix BDHI/, $(CUOBJS_BDHI))

OBJS = $(addprefix Interactor/, $(OBJS_INTERACTOR)) $(addprefix Integrator/, $(OBJS_INTEGRATOR))
OBJS += $(addprefix Measurable/, $(OBJS_MEASURABLE))
OBJS +=  main.o utils/utils.o misc/Potential.o  Driver/Driver.o  Driver/SimulationScript.o $(CUSTOM_OBJS)  misc/Diagonalize.o $(INPUT_FILE_OBJ) 

CUOBJS = $(addprefix Interactor/, $(CUOBJS_INTERACTOR)) $(addprefix Integrator/, $(CUOBJS_INTEGRATOR)) $(CUSTOM_CUOBJS) misc/LanczosAlgorithm.o

THIRD_PARTY_OBJSS = $(addprefix third_party/, $(THIRD_PARTY_OBJS))



ifdef EXPERIMENTAL
OBJS += $(addprefix Interactor/Experimental/, $(OBJS_INTERACTOR_EXPERIMENTAL))
CUOBJS += $(addprefix Interactor/Experimental/, $(CUOBJS_INTERACTOR_EXPERIMENTAL))
endif

all: main

$(THIRD_PARTY_OBJSS):
	$(MAKE) -C third_party/


main:  $(OBJS) $(THIRD_PARTY_OBJSS) $(FORTRAN_OBJS)  $(CUOBJS)
	$(NVCC) $(NVFLAGS) $(LDFLAGS) $(GENCODE_FLAGS) $(INCLUDEFLAGS) -dlink $^  $(LDLIBS) -o gpu.o
	$(CXX) $(CXXFLAGS)  gpu.o $^  $(LDFLAGS) $(LDLIBS) -o uammd



%.o:%.cu
	$(NVCC) $(NVFLAGS) -dc $< -o $@
%.o:%.cpp
	$(NVCC) $(NVFLAGS) -x cu -dc $< -o $@
%.o:%.f90
	gfortran -O3 -c $< -o $@


clean:
	rm -f $(CUOBJS) $(OBJS) $(THIRD_PARTY_OBJSS) $(FORTRAN_OBJS) uammd

cleanCPU:
	rm -f $(OBJS) main
cleanGPU:
	rm -f $(CUOBJS) main
redo: clean all



