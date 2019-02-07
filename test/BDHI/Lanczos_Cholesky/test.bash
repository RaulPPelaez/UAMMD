#Checks the correctness of a Brownian hydrodynamics simulation, see process.cpp for more info.
#Checks that the RPY tensor is being correctly applied from  a simulation with particles of two different sizes

make
g++ -std=c++11 -O3 process.cpp -o process


cat<<EOF>data.main
N              5000
boxSize	       4 4 4
radius_min     0.38173
radius_max     1.89538
outfile	       /dev/stdout
temperature    0
viscosity      1.2131
dt	       10
tolerance      1e-8
nsteps	       100
printSteps     1     #This should be one for this test to make sense
mode	       Lanczos
EOF

error=$(./bdhi 2> lanczos.log | tee pos.dat |
	    ./process |
	    tee lanczos.deviation_from_theory |
	    awk '$2>maxf{maxf=$2}$3>maxg{maxg=$3}END{print maxf*1, maxg*1}')

if echo $error | awk '$1>1e-7 || $2>1e-7 {exit 1}'
then    echo "LANCZOS PASSED $error"    
else    echo "LANCZOS FAILED $error"
fi

sed -i 's+Lanczos+Cholesky+g' data.main

error=$(./bdhi 2> cholesky.log |
	    ./process |
	    tee cholesky.deviation_from_theory |
	    awk '$2>maxf{maxf=$2}$3>maxg{maxg=$3}END{print maxf*1, maxg*1}')

if echo $error | awk '$1>1e-7 || $2>1e-7 {exit 1}'
then    echo "CHOLESKY PASSED $error"    
else    echo "CHOLESKY FAILED $error"
fi

make clean
rm -f process 
