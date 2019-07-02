










make



T=1.23234
vis=2.23536
phi=1
a=3.9782
tol=1e-4

function checkSelfDiffusion {
    scheme=$1
    for l in $( seq 16 8 512)
    do
	L=$(echo 1  | awk '{print '$l'*'$a';}')
	
	N=$(echo 1 | awk '{print int('$phi'*'$L'**2/(3.1415*'$a'*'$a') +0.5)}')

	Nsteps=1000
	dt=0.0001
	if [ "$scheme" == "quasi2D" ] ; then
	    D0=$(echo 1 | awk '{pi=atan2(0,-1); print '$T'/(6*pi*'$vis'*'$a')*(1/(1+4.41/'$l')) }')
	elif [ "$scheme" == "true2D" ] ; then
	    D0=$(echo 1 | awk '{pi=atan2(0,-1); print '$T'/(4*pi*'$vis')*log('$l'/(3.71))}')
	else
	    >&2 echo "Unknown scheme!"
	    exit
	fi

	msdtmp=$(mktemp)
	
	cat<<EOF | ./q2D /dev/stdin 2> /dev/null | fastmsd -N $N -Nsteps $Nsteps -precision float 2> /dev/null | awk '{print $1*'$dt', ($2+$3)/(4*'$D0')}' > $msdtmp

		scheme 	       	     	$scheme
		boxSize			$L $L
		cells 			-1 -1
		numberSteps		$Nsteps
		printSteps	       	1
		dt			$dt
		relaxSteps		0
		viscosity		$vis
		temperature		$T
		hydrodynamicRadius	$a
		loadParticles 		0	
		numberParticles		$N
		tolerance		$tol
		
		output			/dev/stdout 
EOF


	maxtime=$(echo 1 | awk '{print 5*'$dt'}')
	
	slope=$(cat $msdtmp | awk '$1<='$maxtime'' | awk '{xy += $1*$2; x+=$1; y+=$2; x2+=$1*$1}END{m=(NR*xy - x*y)/(NR*x2-x*x); print m;}')
	rm -f $msdtmp *.ps
	
	echo $l $slope;
    done | awk '{print $1, (1.0-$2); fflush(stdout)}'
}

checkSelfDiffusion "true2D"   > selfDiffusionDeviation.true2D

checkSelfDiffusion "quasi2D"   > selfDiffusionDeviation.quasi2D





