function createDataMain {

cat<<EOF > data.main
scheme $scheme
acceptanceRatio 0.5
potential repulsive
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
cutOff  $rcut
sigma   $sigma
epsilon 0.1
shiftLJ 0
U0 $U0
b 0.1
d 1
r_m $r_m
p $p
#readFile init.pos
useElectrostatics
tolerance 1e-6
permitivity $permitivity
gw $gw
split $split
EOF
}

function optimizeSplit {
    relaxSteps=0
    printSteps=-1
    nsteps=1000
    sp=0
    fpsPrev=0
    for split in $(seq 0.025 0.025 1)
    do
	createDataMain
	fps=$(./poisson 2>&1 | tee log | grep FPS | cut -f2 -d:)
	if $(echo $? | awk '{exit(!$1)}');
	then
	    echo "UAMMD failed" >/dev/stderr
	    tail -5 log >/dev/stderr
	    continue
	fi
	echo $(cat data.main | grep split) "FPS:" $fps >/dev/stderr
	if ! echo $fps $fpsPrev | awk '$1<$2{exit 1}'
	then
	    break
	fi
	sp=$split;
	fpsPrev=$fps
    done 
    echo $sp
}

function rdfTheory {
    debyeLength=$(echo 1 | awk '{print sqrt('$permitivity'/(2*'$Molarity'/0.001*'$r_ion'**3*6.022e23));}');
    dr=$(echo $1 $2 | awk '{print $1/$2}');
    g++ tools/dhordf.cpp -o tools/dhordf
    seq 0 $dr $1 | ./tools/dhordf $3 $U0 $permitivity $r_m $p $gw $debyeLength $sigma 
}

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


T=1.0
Molarity=0.05
N=4096

vis=$(echo 1 | awk '{pi=atan2(0,-1); printf("%.14g\n", 1/(6.0*pi))}')
rh=1
r_ion=2e-10
L=$(echo $N $Molarity $r_ion | awk '{print (0.5*$1/(6.022e23*$2)*0.001)^(1/3.0)/$3}')
D0=$(echo $T $vis $rh | awk '{pi=atan2(0,-1);printf("%.14g\n",$1/(6*pi*$2*$3))}')
tauD0=$(echo $D0 $rh | awk '{print $2*$2/$1}')
simulationTime=1000
printTime=1

sigma=$(echo $rh | awk '{print 2*$1}')
p=2
rcut=$(echo $sigma $p | awk '{print $1*2**(1.0/$2)}')
gw=0.25
permitivity=2.219205956747028e-02
r_m=$rh
U0=2.232749391727494e-01
split=0.1
make
rdfTheory 15 2000 1 > rdf.pp.theo
rdfTheory 15 2000 -1 > rdf.pm.theo

for scheme in  AdamsBashforth Leimkuhler EulerMaruyama  MidPoint
do
    for dt in  2e-2 1e-2 1e-3 # 1e-4
    do
	echo "Doing $scheme with dt $dt"
	split=$(optimizeSplit)
	echo "Using split $split"
	#exit
	relaxSteps=$(echo $dt $tauD0 | awk '{print int(30*$2/$1+0.5)}')
	nsteps=$(echo $dt $tauD0 $simulationTime | awk '{print int($2*$3/$1+0.5)}')
	printSteps=$(echo $dt $tauD0 $printTime | awk '{print int($2*$3/$1+0.5)}')	
	createDataMain	
	NstepsInFile=$(echo $nsteps $printSteps | awk '{print int($1/$2)}')
	./poisson data.main 2>log | tee out.pos | awk '{print $1, $2, $3 ,$5}' | 
	    tools/rdf -useTypes -N $N -Nsnapshots $NstepsInFile -nbins 50 -rcut 15 -L $L > rdf.dt$dt.$scheme.tol6.dat
    done
done

mkdir -p tmp
cd tmp
for i in $(ls rdf.dt*);
do
    cat $i |
	awk '/#/{f++}f==2&&!/#/{print $1, $2, $3}' > "BD $(echo $i | cut -d. -f3) $(echo $i | cut -d. -f2)";
done

xmgrace ../rdf.pm.theo -type xydy BD* &
cd ..

