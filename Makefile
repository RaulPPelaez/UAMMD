
INPUT_FILE=src/Driver/SimulationConfig.cpp
all:
	$(MAKE) -C src/ all -j5 INPUT_FILE=../$(INPUT_FILE)
	mkdir -p bin
	mv src/uammd  bin/uammd
clean:
	rm -f bin/uammd
	$(MAKE) -C src/ clean
redo:
	$(MAKE) -C src/ redo 
