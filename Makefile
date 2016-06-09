
all:
	$(MAKE) -C src/ all -j5
	mkdir -p bin
	mv src/main  bin/cudamd
clean:
	rm -f bin/cudamd
	$(MAKE) -C src/ clean
redo:
	$(MAKE) -C src/ redo 
