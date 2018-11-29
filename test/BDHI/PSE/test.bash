
if ! which gracebat > /dev/null 2>&1
then
    echo "PSE TEST FAILED: I need gracebat to plot results!, install grace"
    exit 1
fi


if ! which gfortran > /dev/null 2>&1
then
    echo "PSE TEST FAILED: I need gfortran to compute theory!"
    exit 1
fi

#Any number should produce equivalent results (except for the time step). The resulting plots will be adimensional.
temperature=2.0
viscosity=0.3
hydrodynamicRadius=0.1
make clean pse

for tolerance in 1e-3 # 1e-1 5e-2 1e-2 5e-3 1e-3 1e-7
do    

    resultsFolder=results_$tolerance
    figuresFolder=figures_$tolerance
    mkdir -p $resultsFolder

    ./pse selfMobility $temperature $viscosity $hydrodynamicRadius $tolerance > uammd.selfMobility.log 2>&1 

    gracebat selfMobility_pullForce.psi*.test -par tools/selfMobility.par -hdevice EPS -printfile selfMobility.eps

    
    L=$(echo $hydrodynamicRadius | awk '{print 64*$1}')
    gfortran -O3 tools/ewaldSumRPY.f90 -o ewaldSumRPY; echo "$hydrodynamicRadius $L $L $L" | ./ewaldSumRPY | awk '{print $1/'$hydrodynamicRadius', $2}' > theo; rm ewaldSumRPY


    ./pse pairMobility $temperature $viscosity $hydrodynamicRadius $tolerance > uammd.pairMobility.log 2>&1 

    gracebat theo pairMobility_pullForce.psi*.test -par tools/pairMobility.par -hdevice EPS -printfile pairMobility.eps
    #    rm theo
    mv theo pairMobility.theo

    mv pairMobility.theo selfMobility_pullForce*.test pairMobility_pullForce*.test uammd*log $resultsFolder/

    ./pse selfDiffusion $temperature $viscosity $hydrodynamicRadius $tolerance  > uammd.selfDiffusion.log 2>&1 
    g++ -O3 -std=c++11 tools/msd.cpp tools/Timer.cpp -o msd
    L=$(echo $hydrodynamicRadius | awk '{print 64*$1}')
    for i in $(ls pos.noise*.test)
    do
	dt=$(echo $i | awk -F dt '{print $2}' | awk -F .test '{print $1}')
	./msd -N 4096 -Nsteps 1500 $i 2>/dev/null |
	    awk '{MSD0=(2*'$temperature'*$1*'$dt'/(6*3.1415*'$viscosity'*'$hydrodynamicRadius')*(1-2.837297*'$hydrodynamicRadius'/'$L'));
	      print $1*'$dt', $2/MSD0, $3/MSD0, $4/MSD0}' |
	    head -10 | #Compute mean and std
	    tee $i.msd |
	    awk '{c++; 
	      d=$2-mx; mx += d/c; d2 = $2-mx; x2 += d*d2;
	      d=$3-my; my += d/c; d2 = $3-my; y2 += d*d2;
	      d=$4-mz; mz += d/c; d2 = $4-mz; z2 += d*d2;
	     }
	     END{
	     print mx, my, mz, x2/c, y2/c, z2/c;
 	     }'	 > $i.diffusion & 
	
	if [ $(jobs -p | wc -l) -ge "4" ] 
	then
	    wait
	fi
	
    done
    wait
    rm msd
    for i in $(ls *.diff* | cut -d. -f3,4 | sed 's+psi++')
    do
	awk '{printf "%e %e %e \n", '$i', $1, $4 >> "diff.x"; 
    	  printf "%e %e %e \n", '$i', $2, $5 >> "diff.y";
	  printf "%e %e %e \n", '$i', $3, $6 >> "diff.z";}' pos.noise.psi$i.*diffusion
    done


    gracebat -type xydy diff.x diff.y diff.z -par tools/selfDiffusion.par -hdevice EPS -printfile selfDiffusion.eps


    paste diff.{x,y,z} | awk '{print $1, $2, $3, $5,$6, $8, $9}' > $resultsFolder/selfDiffusion.dat
    mv diff.{x,y,z} $resultsFolder/
    mv pos.noise* $resultsFolder/

    ./pse noiseVariance $temperature $viscosity $hydrodynamicRadius $tolerance > uammd.noiseVariance.log 2>&1 
    cat noiseVariance.test |
	awk '{D0 = $5; var=2*D0; print $1, $2/var, $3/var, $4/var}' > kk


    gracebat -nxy kk -par tools/noiseVariance.par -hdevice EPS -printfile noiseVariance.eps

    rm kk
    mv noiseVariance.test $resultsFolder/


    mkdir -p $figuresFolder

    mv *.eps $figuresFolder
    mv uammd.*log $resultsFolder/


done
