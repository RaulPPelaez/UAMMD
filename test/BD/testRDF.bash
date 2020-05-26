mkdir -p tools
cd tools
if ! test -f rdf
then
    git clone https://github.com/raulppelaez/RadialDistributionFunction
    cd RadialDistributionFunction
    mkdir build && cd build && cmake .. && make -j4; cd ..
    cp build/bin/rdf ..
    cd ..
    rm -rf RadialDistributionFunction
fi
cd ..



T=1
density=0.5
N=8192

vis=$(echo 1 | awk '{pi=atan2(0,-1); printf("%.14g\n", 1/(6.0*pi))}')
rh=1
L=$(echo $N $density | awk '{print ($1/$2)^(1/3.0)}')
simulationTime=50
printTime=1

#make
for scheme in EulerMaruyama  MidPoint
do
    for dt in  5e-3 1e-2 1e-3 1e-4 1e-5
    do
	relaxSteps=$(echo $dt | awk '{print int(10/$1+0.5)}')
	nsteps=$(echo $dt $simulationTime | awk '{print int($2/$1+0.5)}')
	printSteps=$(echo $dt $printTime | awk '{print int($2/$1+0.5)}')
	cat<<EOF > data.main
scheme $scheme
potential soft
boxSize $L $L $L
numberSteps $nsteps
printSteps $printSteps
relaxSteps $relaxSteps
dt         $dt
numberParticles $N
temperature    $T
viscosity   $vis
hydrodynamicRadius $rh
outfile /dev/stdout
cutOff  2.5
sigma   1
epsilon 1
shiftLJ 0
U0 1
b 0.1
d 1
EOF

	NstepsInFile=$(echo $nsteps $printSteps | awk '{print int($1/$2)}')
	./bd data.main 2>log | tee out.pos | 
	    tools/rdf -N $N -Nsnapshots $NstepsInFile -nbins 200 -rcut 4 -L $L > rdf.dt$dt.$scheme.dat
    done
done
