CXX=g++
CC=$(CXX)
NVCC=nvcc

#EXPERIMENTAL=1  #Uncomment to compile with experimental modules

###############################Put here your system configuration:#####################################

#the code will look here for the cuda and thrust sources
CUDA_INCLUDE=/usr/local/cuda/include
#The place where the cuda libs are, libcuda.so, libcudart, libcurand..
CUDA_LIBS=/usr/local/cuda/lib64
#The code will look for a folder named cub here, you can create a symlink if you have a folder i.e. cub-1.5.2
CUB_INCLUDE=/usr/local/include

ARCH ?= 52

#You can place your custom files either here or below in the diferent OBJS_*/CUOBJS_* variables. Dont forget the folders if you put it here!
#These will be compiled with $(CXX)
CUSTOM_OBJS=

#These ones with $(NVCC)
CUSTOM_CUOBJS=

FORTRAN_OBJS =

#############################################################################################################################

#DEBUG= -g

CFLAGS= -g -pthread -std=c++11 -O3  -I$(CUDA_INCLUDE) -march=native -pthread -I$(CUB_INCLUDE) -funroll-loops -m64 -I./ $(DEBUG) -Wall

ifdef EXPERIMENTAL
CFLAGS += -DEXPERIMENTAL
endif


ifdef EXPERIMENTAL
CFLAGS += -DEXPERIMENTAL
endif   


CXXFLAGS=$(CFLAGS) 


$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ifdef DEBUG
DEBUGGPU= -G
endif
NVFLAGS= -g $(DEBUGGPU) -std=c++11 -D_FORCE_INLINES -O3 -ccbin=g++ -lineinfo -m64 $(GENCODE_FLAGS) -src-in-ptx  -I./  -Xptxas -dlcm=ca

LDFLAGS= -L$(CUDA_LIBS)
LDLIBS= -lcuda -lcudart -lpthread -lcurand -lcusolver -lcublas -lgfortran


OBJS_INTERACTOR_EXPERIMENTAL= PairForcesAlt.o
CUOBJS_INTERACTOR_EXPERIMENTAL= PairForcesAltGPU.o 


THIRD_PARTY_OBJS= bravais.o

OBJS_INTERACTOR= Interactor.o PairForces.o PairForcesDPD.o BondedForces.o NBodyForces.o ExternalForces.o 

OBJS_INTEGRATOR= Integrator.o VerletNVE.o VerletNVT.o BrownianEulerMaruyama.o BrownianHydrodynamicsEulerMaruyama.o
OBJS_MEASURABLE= Measurable.o EnergyMeasure.o


CUOBJS_INTERACTOR= PairForcesGPU.o BondedForcesGPU.o NBodyForcesGPU.o ExternalForcesGPU.o
CUOBJS_INTEGRATOR= VerletNVEGPU.o VerletNVTGPU.o BrownianEulerMaruyamaGPU.o BrownianHydrodynamicsEulerMaruyamaGPU.o 

OBJS = $(addprefix Interactor/, $(OBJS_INTERACTOR)) $(addprefix Integrator/, $(OBJS_INTEGRATOR))
OBJS += $(addprefix Measurable/, $(OBJS_MEASURABLE))
OBJS += utils/utils.o misc/Potential.o Driver/Driver.o Driver/SimulationConfig.o $(CUSTOM_OBJS)

CUOBJS = $(addprefix Interactor/, $(CUOBJS_INTERACTOR)) $(addprefix Integrator/, $(CUOBJS_INTEGRATOR)) $(CUSTOM_CUOBJS)

THIRD_PARTY_OBJSS = $(addprefix third_party/, $(THIRD_PARTY_OBJS))



ifdef EXPERIMENTAL
OBJS += $(addprefix Interactor/Experimental/, $(OBJS_INTERACTOR_EXPERIMENTAL))
CUOBJS += $(addprefix Interactor/Experimental/, $(CUOBJS_INTERACTOR_EXPERIMENTAL))
endif

all: main

$(THIRD_PARTY_OBJSS):
	$(MAKE) -C third_party/		


main:  $(OBJS) $(CUOBJS) $(THIRD_PARTY_OBJSS) $(FORTRAN_OBJS)



%.o:%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
%.o:%.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@
%.o:%.f90
	gfortran -O3 -c $< -o $@


clean:
	rm -f $(CUOBJS) $(OBJS) $(THIRD_PARTY_OBJSS) $(FORTRAN_OBJS) main

cleanCPU:
	rm -f $(OBJS) main	
cleanGPU:
	rm -f $(CUOBJS) main	
redo: clean all



