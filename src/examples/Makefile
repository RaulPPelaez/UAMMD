


all: wca lj lj_types pse bd

wca:
	make -j4 -C ../ INPUT_FILE=examples/WCA.cpp
	mv ../uammd WCA
lj:
	make -j4 -C ../ INPUT_FILE=examples/LJ.cpp
	mv ../uammd LJ
lj_types:
	make -j4 -C ../ INPUT_FILE=examples/LJ_types.cpp
	mv ../uammd LJ_types
bd:
	make -j4 -C ../ INPUT_FILE=examples/BD.cpp
	mv ../uammd BD
pse:
	make -j4 -C ../ INPUT_FILE=examples/PSE_test.cpp
	mv ../uammd PSE_test

clean:
	rm -f  WCA LJ LJ_types BD PSE_test *.o
