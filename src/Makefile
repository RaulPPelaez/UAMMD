CXX=g++
CC=$(CXX)
NVCC=nvcc

#  -Ofast -march=native -pipe -funroll-loops
CFLAGS= -pthread -std=c++11 -O3  -I/usr/local/cuda/include/ -march=native -lpthread -I/usr/local/include -funroll-loops -m64 -I./

#CFLAGS= -std=c++11 -O3  -I/usr/local/cuda-7.5/include/ -march=native -g
CXXFLAGS=$(CFLAGS) 

NVFLAGS= -D_FORCE_INLINES -O3 -ccbin=g++ -lineinfo -m64 -arch=sm_52 -restrict -src-in-ptx  -I./

#LDFLAGS= -L/usr/local/cuda-7.5/lib64 -L/usr/lib64/nvidia
LDFLAGS= -L/usr/local/cuda-7.5/lib64 
LDLIBS= -lcuda -lcudart -lpthread -lcurand


OBJS_INTERACTOR= Interactor.o PairForces.o BondedForces.o
OBJS_INTEGRATOR= Integrator.o TwoStepVelVerlet.o BrownianEulerMaruyama.o
OBJS_MEASURABLE= Measurable.o EnergyMeasure.o


CUOBJS_INTERACTOR= PairForcesGPU.o  BondedForcesGPU.o 
CUOBJS_INTEGRATOR= TwoStepVelVerletGPU.o BrownianEulerMaruyamaGPU.o

OBJS = $(addprefix Interactor/, $(OBJS_INTERACTOR)) $(addprefix Integrator/, $(OBJS_INTEGRATOR))
OBJS += $(addprefix Measurable/, $(OBJS_MEASURABLE))
OBJS += utils/utils.o misc/Potential.o Driver/Driver.o

CUOBJS = $(addprefix Interactor/, $(CUOBJS_INTERACTOR)) $(addprefix Integrator/, $(CUOBJS_INTEGRATOR))

all: main

main:  $(OBJS) $(CUOBJS)

%.o:%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
%.o:%.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(CUOBJS) $(OBJS) main

redo: clean all



