make -B
dt=0.00001
T=2.8323234 #Temperature, the theoretical estimation of the eq of state only works far from the transition points
sigma=1.12312
epsilon=1.464342
shiftPotential=0 #Shift potential at cut off?
cutOff=2.5  #In units of sigma

numberParticles=5000 #The box will adapt to achieve a certain density

acceptanceRatio=0.5

cd tools
#Download rdf
if ! test  rdf
then
   git clone https://github.com/raulppelaez/RadialDistributionFunction RDF
   cd RDF
   cmake . && make -j2
   cp bin/rdf ..
   cd ..
   rm -rf RDF
fi
cd ..



for scheme in forcebiased  bd
do
    rm -f eq_state.theo eq_state.$scheme
    echo "Running $scheme"
    for dens in $(seq 0.1 0.1 1.0)
    do
	echo "Running $rho: $dens, Temperature: $T"
	cd tools/Lennard_Jones_eqstate/
	bash eos.sh $dens $(echo $T $epsilon | awk '{print $1/$2}') $(echo $cutOff $sigma | awk '{print $1*$2}') $sigma 1 >> ../../eq_state.theo
	cd - >/dev/null
	L=$(echo $numberParticles $dens | awk '{printf "%.13g", ($1/$2)^(1/3.0)*'$sigma'}')    
	nsteps=100000
	printSteps=10000
	relaxSteps=100000
	
	echo "integrator $scheme" > data.main
	echo "acceptanceRatio $acceptanceRatio" >> data.main
	echo "U0 4" >> data.main
	echo "b  0.1" >> data.main
	echo "d  1" >> data.main
	echo "potential LJ" >> data.main
	echo "boxSize $L $L $L " >> data.main
	echo "numberSteps $nsteps" >> data.main
	echo "printSteps $printSteps" >> data.main
	echo "relaxSteps $relaxSteps " >> data.main
	echo "dt $dt" >> data.main
	echo "numberParticles $numberParticles " >> data.main
	echo "sigma $sigma" >> data.main
	echo "epsilon $epsilon " >> data.main
	echo "temperature $T " >> data.main
	echo "outfile rho$dens.pos " >> data.main
	echo "energyOutfile rho$dens.energy " >> data.main
	echo "shiftLJ $shiftPotential" >> data.main
	echo "cutOff $cutOff" >> data.main
	
	./forcebiased > rho$dens.log 2>&1 

	#Compute rdf
	cat rho$dens.pos |
	    ./tools/rdf -N $numberParticles -Nsnapshots $(echo $nsteps $printSteps | awk '{print $1/$2-1}') -L $L -rcut $(echo $L | awk '{print $1/2}') -nbins 500 > rho$dens.rdf

	#Compute pressure
	P=$(bash tools/pressure.sh rho$dens.rdf | awk '{print $1}')

	#Compute energy
	E=$(cat rho$dens.energy | awk '{printf "%.13g", $1+$2}' | awk '{count++; d=$1-mean; mean+=d/count; d2 =$1-mean; m2+=d*d2;}END{printf "%.13g %.13g", mean, m2/count}' | awk '{printf "%.13g", $1/'$epsilon'}')
	
	echo $dens $T $E $P >> eq_state.$scheme
    done
mkdir -p results.$scheme
mv rho* results.$scheme
mv eq_state* results.$scheme

rm data.main

mkdir -p figures

#Plot results
awk '{print $1, $3}' results.$scheme/eq_state.theo > figures/E.theo
awk '{print $1, $4}' results.$scheme/eq_state.theo > figures/P.theo

awk '{print $1, $3}' results.$scheme/eq_state.$scheme > figures/E.$scheme
awk '{print $1, $4}' results.$scheme/eq_state.$scheme > figures/P.$scheme


xmgrace -graph 0 figures/E.theo -graph 0 figures/E.$scheme -graph 1 figures/P.theo -graph 1 figures/P.$scheme -par tools/eq_state.par -hdevice EPS -hardcopy -printfile figures/eq_state.$scheme.eps


paste figures/E.theo figures/E.$scheme | awk '{a=sqrt((($2-$4)/($2+$4))^2);}a>mx{mx=a}END{print "'$scheme': Maximum relative deviation from theory in Energy:", mx}'
paste figures/P.theo figures/P.$scheme | awk '{a=sqrt((($2-$4)/($2+$4))^2);}a>mx{mx=a}END{print "'$scheme': Maximum relative deviation from theory in Pressure:", mx}'
done
