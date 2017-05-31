


all: wca lj lj_types pse bd couette poiseuille pse_logo

wca:
	make -j4 -C ../ INPUT_FILE=examples/WCA.cpp NAME=examples/WCA

lj:
	make -j4 -C ../ INPUT_FILE=examples/LJ.cpp NAME=examples/LJ
lj_types:
	make -j4 -C ../ INPUT_FILE=examples/LJ_types.cpp NAME=examples/LJ_types

bd:
	make -j4 -C ../ INPUT_FILE=examples/BD.cpp NAME=examples/BD

pse:
	make -j4 -C ../ INPUT_FILE=examples/PSE_test.cpp NAME=examples/PSE_test

couette: couette.cpp
	make -j4 -C ../ INPUT_FILE=examples/couette.cpp NAME=examples/couette

poiseuille: poiseuille.cpp
	make -j4 -C ../ INPUT_FILE=examples/poiseuille.cpp NAME=examples/poiseuille

pse_logo: PSE_logo.cpp
	make -j4 -C ../ INPUT_FILE=examples/PSE_logo.cpp NAME=examples/PSE_logo


clean:
	rm -f  WCA LJ LJ_types BD PSE_test couette poiseuille PSE_logo *.o 
