
all:
	$(MAKE) -C src/ all -j5
	mkdir -p bin
	mv src/main src/benchmark bin/
clean:
	rm -f bin/main bin/benchmark
	$(MAKE) -C src/ clean
redo:
	$(MAKE) -C src/ redo 
