
cd tools
#Download rdf
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
#make clean pse

for tolerance in 1e-5 # 1e-1 5e-2 1e-2 5e-3 1e-3 1e-7
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
    mv theo pairMobility.theo

    mv pairMobility.theo selfMobility_pullForce*.test pairMobility_pullForce*.test uammd*log $resultsFolder/

    ./pse selfDiffusion $temperature $viscosity $hydrodynamicRadius $tolerance  > uammd.selfDiffusion.log 2>&1 
    L=$(echo $hydrodynamicRadius | awk '{print 64*$1}')
    for i in $(ls pos.noise*.test)
    do
	dt=$(echo $i | awk -F dt '{print $2}' | awk -F .test '{print $1}')	
	cat $i | ./tools/msd -N 4096 -Nsteps 1500 -device CPU |
	    head -10 |
	    awk '{MSD0=(2*'$temperature'/(6*3.1415*'$viscosity'*'$hydrodynamicRadius')*(1-2.837297*'$hydrodynamicRadius'/'$L'));
	      print $1*'$dt', $2/MSD0, $3/MSD0, $4/MSD0}' |
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
		  printf("%.14g ", 1.0-m, 0);
		}
		printf("\n");
		      }' > $i.diffusion &        
	if [ $(jobs -p | wc -l) -ge "4" ] 
	then
	    wait
	fi	
    done
    wait
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
