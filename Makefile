
all:
	$(MAKE) -C src/ all -j9
	mv src/main bin/main
clean:
	rm -f bin/main
	$(MAKE) -C src/ clean
redo:
	$(MAKE) -C src/ redo 
