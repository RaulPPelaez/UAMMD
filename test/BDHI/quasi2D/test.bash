make 

if ! ls tools/fastmsd >/dev/null 2>&1
then
    mkdir -p tools
    cd tools
    git clone https://github.com/raulppelaez/MeanSquareDisplacement    
    cd MeanSquareDisplacement
    mkdir build
    cd build
    cmake .. && make
    cp bin/msd ../../fastmsd
    cd ../../../
    rm -rf tools/MeanSquareDisplacement
fi


if ! ls tools/rdf >/dev/null 2>&1
then
    mkdir -p tools
    cd tools
    git clone https://github.com/RaulPPelaez/RadialDistributionFunction
    cd RadialDistributionFunction
    mkdir build; cd build; cmake ..; make; cd ..	
    cp build/bin/rdf ../
    cd ..
    cd ..
    rm -rf tools/RadialDistributionFunction
fi


#Random numbers, the test should be independent on the values of these parameters
#Temperature
T=$(head -100 /dev/urandom | cksum | awk '{srand($1);print 1+rand();}')
#Viscosity
vis=$(head -100 /dev/urandom | cksum | awk '{srand($1);print 1+rand();}')
#Hydrodynamic radus
a=$(head -100 /dev/urandom | cksum | awk '{srand($1);print 1+rand();}')

tol=1e-4

function checkSelfDiffusion {
    scheme=$1    
    for l in $(seq 16 32 512)
    do
	Ntry=10
	for i_try in $(seq $Ntry)
	do
	    phi=0.5
	    L=$(echo 1  | awk '{print '$l'*'$a';}')	
	    N=$(echo 1 | awk '{print int('$phi'*'$L'**2/(3.1415*'$a'*'$a') + 0.55)}')
	    Nsteps=1000
	    if [ "$scheme" == "quasi2D" ] ; then
		D0=$(echo 1 | awk '{pi=atan2(0,-1); printf "%.15g\n", '$T'/(6*pi*'$vis'*'$a')*(1/(1+4.41/'$l')) }')
	    elif [ "$scheme" == "true2D" ] ; then
		D0=$(echo 1 | awk '{pi=atan2(0,-1); printf "%.15g\n", '$T'/(4*pi*'$vis')*(log('$l')-1.3105329259115095183)}')
	    else
		>&2 echo "Unknown scheme!"
		exit
	    fi
	    t0=$(echo $a $D0 | awk '{print ($1*$1)/$2}');
	    dt=$(echo $t0 | awk '{print 0.0001*$1}');
	    msdtmp=$(mktemp)
	    cat<<EOF | ./q2D /dev/stdin 2> /dev/null | tools/fastmsd -N $N -Nsteps $Nsteps -precision double 2> /dev/null | awk '{print $1*'$dt', ($2+$3)/(4*'$D0')}' > $msdtmp
		scheme 	       	     	$scheme
		test 			selfDiffusion
		boxSize			$L $L
		cells 			-1 -1
		numberSteps		$Nsteps
		printSteps	       	1
		dt			$dt
		relaxSteps		0
		viscosity		$vis
		temperature		$T
		hydrodynamicRadius	$a
		numberParticles		$N
		tolerance		$tol		
		output			/dev/stdout 
EOF
	    maxtime=$(echo 1 | awk '{print 5*'$dt'}')       
	    slope=$(cat $msdtmp | awk '$1<='$maxtime'' |
			awk '{
			    xy += $1*$2;
		    	    x+=$1; y+=$2;
			    x2+=$1*$1
			  }
			  END{
			    m=(NR*xy - x*y)/(NR*x2-x*x);       
			    print m;
			  }')

	    rm -f $msdtmp *.ps	
	    echo $l $slope;
	done | awk '{printf "%.15g %.15g \n", $1, (1.0-$2);}' |
	    datamash -W groupby 1 mean 2 sstdev 2 count 1 |
	    awk '{printf "%.15g %.15g %.15g\n", $1, $2, $3/sqrt($4);}'
    done 
}

function checkSelfMobility {
    scheme=$1
    for l in $( seq 32 32 1024)
    do
	L=$(echo 1  | awk '{print '$l'*'$a';}')	
	Nsteps=1
	dt=1
	if [ "$scheme" == "quasi2D" ] ; then
	    M0=$(echo 1 | awk '{pi=atan2(0,-1); printf "%.15g\n", 1.0/(6*pi*'$vis'*'$a')*(1/(1+4.41/'$l')) }')
	elif [ "$scheme" == "true2D" ] ; then
	    M0=$(echo 1 | awk '{pi=atan2(0,-1); printf "%.15g\n", 1.0/(4*pi*'$vis')*(log('$l')-1.3105329259115095183)}')
	else
	    >&2 echo "Unknown scheme!"
	    exit
	fi
	pullForce=$(echo 1 | awk '{print '$a'/('$M0'*'$dt');}')
	msdtmp=$(mktemp)	
	cat<<EOF | ./q2D /dev/stdin 2> /dev/null | awk '{printf "%g %.15g\n", '$l', sqrt((1.0-($1)/('$M0'))^2); fflush(stdout);}'
		scheme 	       	     	$scheme
		test 			selfMobility
		boxSize			$L $L
		cells 			-1 -1
		dt			$dt
		viscosity		$vis
		temperature		0
		hydrodynamicRadius	$a
		tolerance		$tol
		F 			$pullForce			
		output			/dev/stdout 
EOF
    done
}


function checkRadialDistributionFunction {
    scheme=$1
    L=128
    N=16384
    Nsteps=1000000
    printSteps=1000
    Nsnapshots=$(echo $Nsteps $printSteps | awk '{print int($1/$2);}')
    dt=0.005
    cat<<EOF | ./q2D /dev/stdin 2> /dev/null | tools/rdf -N $N -Nsnapshots $Nsnapshots -precision double -L $L -nbins 100 -rcut $a -dim 2D 2> /dev/null | awk '{print $1/'$a', $2, $3}' 
		scheme 	       	     	$scheme
		test 			selfDiffusion
		boxSize			$L $L
		cells 			-1 -1
		numberSteps		$Nsteps
		printSteps	       	$printSteps
		dt			$dt
		relaxSteps		0
		viscosity		$vis
		temperature		$T
		hydrodynamicRadius	$a
		numberParticles		$N
		tolerance		$tol		
		output			/dev/stdout 
EOF



}
checkSelfDiffusion "true2D"   > selfDiffusionDeviation.true2D
checkSelfDiffusion "quasi2D"  > selfDiffusionDeviation.quasi2D

checkSelfMobility "true2D"   > selfMobilityDeviation.true2D
checkSelfMobility "quasi2D"   > selfMobilityDeviation.quasi2D

checkRadialDistributionFunction "true2D"   > rdf.true2D
checkRadialDistributionFunction "quasi2D"   > rdf.quasi2D







