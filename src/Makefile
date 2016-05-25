CXX=g++
CC=$(CXX)
NVCC=nvcc

#  -Ofast -march=native -pipe -funroll-loops
CFLAGS= -pthread -std=c++11 -O3  -I/usr/local/cuda-7.5/targets/x86_64-linux/include/ -march=native -lpthread -I/usr/local/include -funroll-loops -m64

#CFLAGS= -std=c++11 -O3  -I/usr/local/cuda-7.5/include/ -march=native -g
CXXFLAGS=$(CFLAGS) 

NVFLAGS= -O3 -ccbin=g++ -lineinfo -m64 -arch=sm_52 -restrict -src-in-ptx 

#LDFLAGS= -L/usr/local/cuda-7.5/lib64 -L/usr/lib64/nvidia
LDFLAGS= -L/usr/local/cuda-7.5/lib64 
LDLIBS=-lcuda -lcudart -lpthread


CUOBJS= InteractorGPU.o
OBJS=main.o Interactor.o utils/utils.o Potential.o

EXE=main

all:main

main:$(CUOBJS) $(OBJS)


%.o:%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
%.o:%.cpp Rlib.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(CUOBJS) $(OBJS) $(EXE)

redo: clean all



