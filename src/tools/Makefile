CXX=g++
CC=$(CXX)


CFLAGS= -std=c++11 -O4 -ffast-math -fno-math-errno -march=native -fopenmp -g 
CXXFLAGS=$(CFLAGS)

LDLIBS= -lgomp -lpthread -pthread

OBJS= pbc.o msd.o gdr.o 
EXE=pbc msd gdr 

all:$(EXE)

msd:Timer.o

coherent_intermediate_scfnc: math_helper.o Timer.o

clean:
	rm -f $(OBJS) $(EXE) *.o

redo: clean all
