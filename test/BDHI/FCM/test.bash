#This script runs several tests that ensure the correct results are provided when using the FCM BDHI module. Several tests exists tackilng the different parts of the algorithm separatedly and jointly.
#Some figures will be produced after this script runs that will make clear if the algorithm works or not.
#Additionally some automatic sanity checks are made. But always look at the figures.
#Expect this script to consume approximately 4GB of GPU memory and take several hours to complete
#You can turn off/on each test in the last lines and change the parameters below, the test should work for any combination of temperature, viscosity... as they should incurr only a change of units. All results should be unitless.
#This script is quite severe and will complain even with seemingly small deviations from theory, so you should judge yourself.
if ! which gracebat > /dev/null 2>&1
then
    echo "FCM TEST FAILED: I need gracebat to plot results!, install grace"
    exit 1
fi

if ! which gnuplot >/dev/null  || !  gnuplot --version | awk '{exit($2<5.0)}'
then
    echo "FCM TEST FAILED: I need at least gnuplot 5.0!"
    exit 1
fi

if ! which gfortran > /dev/null 2>&1
then
    echo "FCM TEST FAILED: I need gfortran to compute theory!"
    exit 1
fi

#Any number should produce equivalent results (except for the time step). The resulting plots will be adimensional.
temperature=1.0
viscosity=2
hydrodynamicRadius=2
tolerance=1e-7
make fcm

resultsFolder=results
figuresFolder=figures

#rm -rf $resultsFolder $figuresFolder
mkdir -p $resultsFolder $figuresFolder



function selfMobilityCubicBox {
    echo "self Mobility cubic box test"
    ./fcm selfMobilityCubicBox 0 $viscosity $hydrodynamicRadius $tolerance  > uammd.selfMobility.log 2>&1 

    gracebat -nxy selfMobilityCubicBox.test -par tools/selfMobility.par -hdevice EPS -printfile selfMobilityCubicBox.eps

    #Compute maximum deviation from theory, all points should be 0
    #Ignore the first values to prevent higher order PBC corrections from giving a false positive
    maxDeviation=$(tail -n+5 selfMobilityCubicBox.test | awk '{for(i=2;i<=NF;i++)print sqrt(($i)**2)}' | datamash -W max 1)

    #The PBC corrections in this test go up to L^-6, so higher order corrections should be expected for small boxes. 
    expectedError=$(head -1 selfMobilityCubicBox.test | awk '{print 2/$1**6+'$tolerance'}')
    if echo $maxDeviation $expectedError | awk '{if($1>$2) exit(0);else exit(1);}'
    then
	echo "SELF MOBILITY TEST FAILED, Maximum deviation: $maxDeviation, tolerance: $tolerance, expected deviation: $expectedError"
    fi

    mv selfMobilityCubicBox.test uammd.selfMobility.log results/
    mv selfMobilityCubicBox.eps figures/
}


function extrapolate {
    #Extrapolates the results of the mobility matrix to L=inf by fitting to a polynomy with gnuplot.
    file=$1
    Minf=$(gnuplot -e 'set fit errorvariables; 
    set fit quiet;
     f(x) = a+b/x+c/x/x+d/x**3 + e/x**4+f/x**5+g/x**6; 
do for [i=2:10] {
     fit f(x) "'$file'" u 1:(column(i)) via a,b,c,d,e,f,g;
      print(a);
}' 2>&1)
    
    echo $Minf
}

function pairMobilityCubicBox {
    echo "Pair mobility cubic box test"
    #Computes 1-M_{\alpha\beta}(\vec{r}, L)/M_{\alpha\beta}(\vec{r}, L=\inf) for two particles with opposing forces, which should converge to 0 for all terms.
    #It computes dist for several distances and uses a random distance between the two particles each time.
    
    ./fcm pairMobilityCubicBox 0 $viscosity $hydrodynamicRadius $tolerance  > uammd.pairMobility.log 2>&1 

    maxDeviation=$(
    for i in $(ls  pairMobilityCubicBox.dist*.test)
    do	
	echo $(extrapolate $i)
    done | tr ' ' '\n' |
	awk '{print sqrt($1**2)}' |
	datamash -W max 1)

    expectedDeviation=1e-4
    if echo $maxDeviation $expectedDeviation | awk '{if($1>$2) exit(0); else exit(1)}'
    then
	echo "PAIR MOBILITY TEST FAILED, Maximum deviation: $maxDeviation, tolerance: $tolerance, expected deviation $expectedDeviation"
    fi
    mv pairMobilityCubicBox.*test uammd.pairMobility.log results/
}



function selfMobility_q2D {
    echo "Self Mobility q2D test"
    ./fcm selfMobility_q2D 0 $viscosity $hydrodynamicRadius $tolerance  > uammd.selfMobility_q2D.log 2>&1 

    gracebat -nxy selfMobility_q2D.test -nxy selfMobility_q2D.theo.test -par tools/selfMobility_q2D.par -hdevice EPS -printfile selfMobility_q2D.eps

    mv selfMobility_q2D.*test uammd.selfMobility_q2D.log results/
    mv selfMobility_q2D.eps figures/
}

function selfDiffusionCubicBox {
    echo "Self diffusion cubic box test"
    ./fcm selfDiffusionCubicBox $temperature $viscosity $hydrodynamicRadius $tolerance  > uammd.selfDiffusion.log 2>&1
    g++ -std=c++11 -O3 tools/msd.cpp tools/Timer.cpp -o msd
    for i in $(ls pos.noise*.test)
    do
	echo "Doing $i" >/dev/stderr
	L=$(echo $i | cut -d. -f3,4 | sed 's+boxSize++')
	dt=$(echo $i | awk -F dt '{print $2}' | awk -F .test '{print $1}')
	#PBC corrections up to sixth order in L
	D0=$(echo 1 | awk '{l=1/'$L';print '$temperature'/(6*3.14159265358979*'$viscosity'*'$hydrodynamicRadius')*(1-2.837297*l+(4/3.0)*3.14159265358979*l**3-27.4*l**6);}')
	
	./msd -N $(grep -n "#" -m 2 $i | cut -d: -f1 | paste -sd" " | awk '{print $2-$1-1}') -Nsteps $(grep -c "#" $i)  $i 2>/dev/null | awk '{print $1*'$dt', $2, $3, $4}'  > $i.msd 
	
	#Fit msd to 2*t*a*D0 in each direction, so "a" should be 1. Weight points with 1/t^4 when fitting.
	slope=$(gnuplot -e 'set fit quiet; f(x) = 2*x*a*'$D0'; do for [i=2:4]{ fit f(x) "'$i'.msd" u 1:(column(i)):($1*$1):(1e-30) errors x,z via a; print a;}' 2>&1)
	
	echo $L $slope
    done | sort -g -k1  > results/selfDiffusionCubicBox.test
	gracebat  -nxy results/selfDiffusionCubicBox.test -par tools/selfDiffusionCubicBox.par -hdevice EPS -printfile figures/selfDiffusionCubicBox.eps

	maxDeviation=$(cat results/selfDiffusionCubicBox.test |
			   awk '{print $2, $3, $4}' |
			   tr ' ' '\n' |
			   awk '{print sqrt((1-$1)**2)}' | 
			   datamash -W max 1 )
	
	echo "Maximum deviation from expected diffusion: $maxDeviation"
	mv pos.noise* uammd.selfDiffusion.log results/
	rm msd

}


function selfDiffusion_q2D {
    echo "Self diffusion q2D test"
    ./fcm selfDiffusion_q2D $temperature $viscosity $hydrodynamicRadius $tolerance  > uammd.selfDiffusion_q2D.log 2>&1
    g++ -std=c++11 -O3 tools/msd.cpp tools/Timer.cpp -o msd
    for i in $(ls pos.noise*.*q2D*test)
    do
	echo "Doing $i" >/dev/stderr
	dt=$(echo $i | awk -F dt '{print $2}' | awk -F .q2D '{print $1}')

	D0=$(echo 1 | awk '{print '$temperature'/(6*3.14159265358979*'$viscosity'*'$hydrodynamicRadius');}')
	N=$(grep -n "#" -m 2 $i | cut -d: -f1 | paste -sd" " | awk '{print $2-$1-1}')
	Nsteps=$(grep -c "#" $i)	
	./msd -N $N -Nsteps $Nsteps $i 2> /dev/null | awk '{print $1*'$dt', $2, $3, $4}'  > $i.msd 
	
	#Fit msd to 2*t*a*D0 in each direction, so "a" should be 1. Weight points with 1/t^4 when fitting.
	slope=$(gnuplot -e 'set fit quiet; f(x) = 2*x*a*'$D0'; do for [i=2:4]{ fit f(x) "'$i'.msd" u 1:(column(i)):($1*$1):(1e-30) errors x,z via a; print a;}' 2>&1)

	L=$(echo $i | cut -d. -f3,4 | sed 's+boxSize++')
	
	echo $L $slope 
    done | sort -g -k1 > results/selfDiffusion_q2D.test
    
    gracebat  -nxy results/selfDiffusion_q2D.test -nxy selfDiffusion_q2D.theo -par tools/selfDiffusion_q2D.par -hdevice EPS -printfile figures/selfDiffusion_q2D.eps
	
    mv pos.noise* uammd.selfDiffusion_q2D.log selfDiffusion_q2D.theo  results/
    rm msd
}

function noiseVariance {
    echo "Noise variance test"
    ./fcm noiseVariance $temperature $viscosity $hydrodynamicRadius $tolerance > uammd.noiseVariance.log 2>&1 
    gracebat -nxy noiseVariance.test -par tools/noiseVariance.par -hdevice EPS -printfile figures/noiseVariance.eps
    mv noiseVariance.test uammd.noiseVariance.log $resultsFolder/
}

pairMobilityCubicBox
#./fcm pairMobility_q2D 0 $viscosity $hydrodynamicRadius $tolerance  > uammd.pairMobility_q2D.log 2>&1 
selfMobilityCubicBox
selfMobility_q2D
selfDiffusionCubicBox
selfDiffusion_q2D
noiseVariance


rm -f fit.log
