make -C ../../examples/generic_md/
cp ../../examples/generic_md/generic langevin
dt=0.0005
T=3 #Temperature, the theoretical estimation of the eq of state only works far from the transition points
sigma=1
epsilon=1
cutOff=2.5  #Reduced units, note that theory only works for rcut=2.5

numberParticles=16384 #The box will adapt to achieve a certain density

mkdir -p tools
cd tools
if ! test -f rdf
then
    git clone https://github.com/raulppelaez/RadialDistributionFunction
    if ! (cd RadialDistributionFunction && mkdir build && cd build && cmake .. && make -j4)
    then
	echo "ERROR When compiling RadialDistributionFunction, check inside tools/ and ensure tools/rdf exists before continuing" > /dev/stderr
	exit 1
    fi
    cp RadialDistributionFunction/build/bin/rdf .
    rm -rf RadialDistributionFunction
fi
cd ..

rm -f eq_state.theo eq_state.langevin

for dens in $(seq 0.1 0.05 1.0)
do
    echo "Running $rho: $dens, Temperature: $T"
    cd tools #/Lennard_Jones_eqstate/
    bash eos.sh $dens $(echo $T $epsilon | awk '{print $1/$2}')  >> ../eq_state.theo
    cd -
    L=$(echo $numberParticles $dens | awk '{printf "%.13g", ($1/$2)^(1/3.0)*'$sigma'}')
    
    nsteps=200000
    printSteps=250
    relaxSteps=40000
    echo "integrator VerletNVT" > data.main
    echo "L $L $L $L " >> data.main
    echo "numberSteps $nsteps" >> data.main
    echo "printSteps $printSteps" >> data.main
    echo "relaxSteps $relaxSteps " >> data.main
    echo "dt $dt" >> data.main
    echo "numberParticles $numberParticles " >> data.main
    echo "sigma $sigma" >> data.main
    echo "epsilon $epsilon " >> data.main
    echo "temperature $T " >> data.main
    echo "outfile rho$dens.pos " >> data.main
    echo "outfileEnergy rho$dens.energy " >> data.main
    echo "cutOff $cutOff" >> data.main
    echo "friction 1.0" >> data.main
    ./langevin > rho$dens.log 2>&1 

    #Compute rdf
    cat rho$dens.pos |
	./tools/rdf -N $numberParticles -Nsnapshots $(echo $nsteps $printSteps | awk '{print $1/$2-1}') -L $L -rcut $cutOff -nbins 1000 > rho$dens.rdf

    #Compute pressure
    P=$(bash tools/pressure.sh rho$dens.rdf | awk '{print $1}')

    #Compute energy
    E=$(cat rho$dens.energy |
	    grep Total |
	    awk '{print $4/'$numberParticles'}' |
	    awk '{count++; d=$1-mean; mean+=d/count; d2 =$1-mean; m2+=d*d2;}END{printf "%.13g %.13g", mean, m2/count}' | awk '{printf "%.13g", $1/'$epsilon'}')
   
    echo $dens $(echo $T $epsilon | awk '{print $1/$2}') $E $P >> eq_state.langevin 
        
done

mkdir -p results
mv rho* results
mv eq_state* results

rm data.main

mkdir -p figures

#Plot results
awk '{print $1, $3}' results/eq_state.theo > figures/E.theo
awk '{print $1, $4}' results/eq_state.theo > figures/P.theo

awk '{print $1, $3}' results/eq_state.langevin > figures/E.langevin
awk '{print $1, $4}' results/eq_state.langevin > figures/P.langevin


xmgrace -graph 0 figures/E.theo -graph 0 figures/E.langevin -graph 1 figures/P.theo -graph 1 figures/P.langevin -par tools/eq_state.par -hdevice EPS -hardcopy -printfile figures/eq_state.eps


paste figures/E.theo figures/E.langevin | awk '{a=sqrt((1.0-$2/$4)^2);}a>mx{mx=a}END{print "Maximum relative deviation from theory in Energy:", mx}'
paste figures/P.theo figures/P.langevin | awk '{a=sqrt((1.0-$2/$4)^2);}a>mx{mx=a}END{print "Maximum relative deviation from theory in Pressure:", mx}'
