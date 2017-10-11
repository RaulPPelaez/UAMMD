CXX=g++
CC=$(CXX)


CFLAGS= -std=c++11 -O4 -ffast-math -fno-math-errno -march=native -fopenmp -g 
CXXFLAGS=$(CFLAGS)

LDLIBS= -lgomp -lpthread -pthread

OBJS= pbc.o msd.o gdr.o 
EXE=pbc msd gdr Elastic_Network_Model fft

all:$(EXE)

msd:Timer.o

fft:
        nvcc -std=c++11 -O3 -arch=sm_52 --expt-extended-lambda fft.cu -lcufft -o fft
clean:
	rm -f $(OBJS) $(EXE) *.o

redo: clean all
