mkdir -p tools
cd tools
if ! test -f msd
then
   git clone https://github.com/raulppelaez/MeanSquareDisplacement
   cd MeanSquareDisplacement
   mkdir build && cd build && cmake .. && make -j4; cd ..
   cp build/bin/msd ..
   cd ..
   rm -rf MeanSquareDisplacement
fi
cd ..
#scheme=EulerMaruyama
scheme=MidPoint
L=32
N=16384
vis=$(echo 1 | awk '{pi=atan2(0,-1); printf("%.14g\n", 1/(6.0*pi))}')
rh=1
T=1
D0=$(echo $T $vis $rh | awk '{pi=atan2(0,-1);printf("%.14g\n",$1/(6*pi*$2*$3))}')

relaxSteps=-1
nsteps=1000
printSteps=1
dt=1e-6

cat<<EOF > data.main
scheme $scheme
potential LJ
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
EOF

make
Nsteps=$(echo $nsteps $printSteps | awk '{print int($1/$2)}')
./bd data.main 2>log |
    tools/msd -N $N -Nsteps $Nsteps |
    head -5 |
    awk '{
	t+=$1;
	t2+=$1*$1;
	nf=NF;
	for(i=2; i<=nf; i++){
	  xt[i] += $1*$i;
	  x[i]+=$i;
	}
	}
	END{
	for(i=2; i<=nf; i++){
          m=(NR*xt[i] - x[i]*t)/(NR*t2-t*t);
	  printf("Relative deviation from expected self Diffusion coefficient: %.14g ", 1.0-m/(2*'$dt'*'$printSteps'*'$D0'));
	}
	printf("\n");}'





