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


OBJS_INTERACTOR= Interactor/Interactor.o Interactor/PairForces.o
OBS_INTEGRATOR= Integrator/Integrator.o Integrator/TwoStepVelVerlet.o Integrator/BrownianEulerMaruyama.o

CUOBJS= Interactor/PairForcesGPU.o Integrator/TwoStepVelVerletGPU.o Integrator/BrownianEulerMaruyamaGPU.o

OBJS= $(OBJS_INTERACTOR) $(OBS_INTEGRATOR)  utils/utils.o misc/Potential.o Driver/Driver.o


all: main

main:  $(OBJS) $(CUOBJS)

%.o:%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
%.o:%.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(CUOBJS) $(OBJS) main

redo: clean all



