CXX=g++
CC=$(CXX)
NVCC=nvcc


###############################Put here your system configuration:#####################################

#the code will look here for the cuda and thrust sources
CUDA_INCLUDE=/usr/local/cuda/include
#The place where the cuda libs are, libcuda.so, libcudart, libcurand..
CUDA_LIBS=/usr/local/cuda/lib64
#The code will look for a folder named cub here, you can create a symlink if you have a folder i.e. cub-1.5.2
CUB_INCLUDE=/usr/local/include


#You can place your custom files either here or below in the diferent OBJS_*/CUOBJS_* variables. Dont forget the folders if you put it here!
#These will be compiled with $(CXX)
CUSTOM_OBJS=

#These ones with $(NVCC)
CUSTOM_CUOBJS=


#############################################################################################################################


CFLAGS= -pthread -std=c++11 -O3  -I$(CUDA_INCLUDE) -march=native -lpthread -I$(CUB_INCLUDE) -funroll-loops -m64 -I./ -g

CXXFLAGS=$(CFLAGS) 

NVFLAGS= -g -D_FORCE_INLINES -O3 -ccbin=g++ -lineinfo -m64 -arch=sm_52 -restrict -src-in-ptx  -I./

LDFLAGS= -L$(CUDA_LIBS)
LDLIBS= -lcuda -lcudart -lpthread -lcurand -lcusolver -lcublas

OBJS_INTERACTOR= Interactor.o PairForces.o PairForcesDPD.o BondedForces.o NBodyForces.o ExternalForces.o 
OBJS_INTEGRATOR= Integrator.o VerletNVE.o VerletNVT.o BrownianEulerMaruyama.o BrownianHydrodynamicsEulerMaruyama.o
OBJS_MEASURABLE= Measurable.o EnergyMeasure.o


CUOBJS_INTERACTOR= PairForcesGPU.o BondedForcesGPU.o NBodyForcesGPU.o ExternalForcesGPU.o
CUOBJS_INTEGRATOR= VerletNVEGPU.o VerletNVTGPU.o BrownianEulerMaruyamaGPU.o BrownianHydrodynamicsEulerMaruyamaGPU.o 

OBJS = $(addprefix Interactor/, $(OBJS_INTERACTOR)) $(addprefix Integrator/, $(OBJS_INTEGRATOR))
OBJS += $(addprefix Measurable/, $(OBJS_MEASURABLE))
OBJS += utils/utils.o misc/Potential.o Driver/Driver.o Driver/SimulationConfig.o $(CUSTOM_OBJS)

CUOBJS = $(addprefix Interactor/, $(CUOBJS_INTERACTOR)) $(addprefix Integrator/, $(CUOBJS_INTEGRATOR)) $(CUSTOM_CUOBJS)

all: main

main:  $(OBJS) $(CUOBJS)

%.o:%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
%.o:%.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(CUOBJS) $(OBJS) main

cleanCPU:
	rm -f $(OBJS) main	
cleanGPU:
	rm -f $(CUOBJS) main	
redo: clean all



