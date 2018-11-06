
if ! which gracebat > /dev/null 2>&1
then
    echo "FIB TEST FAILED: I need gracebat to plot results!, install grace"
    exit 1
fi


if ! which gfortran > /dev/null 2>&1
then
    echo "FIB TEST FAILED: I need gfortran to compute theory!"
    exit 1
fi

#Any number should produce equivalent results (except for the time step). The resulting plots will be adimensional.
temperature=2.0
viscosity=0.25
hydrodynamicRadius=2
scheme="simple_midpoint"
make fib

resultsFolder=results
figuresFolder=figures
mkdir -p $resultsFolder

./fib selfMobility $temperature $viscosity $hydrodynamicRadius $scheme > uammd.selfMobility.log 2>&1 

gracebat selfMobility_pullForce.test -par tools/selfMobility.par -hdevice EPS -printfile selfMobility.eps


./fib pairMobility $temperature $viscosity $hydrodynamicRadius $scheme > uammd.pairMobility.log 2>&1 

L=$(echo $hydrodynamicRadius | awk '{print 64*$1/0.91}') #I want L=64 with rh=0.91 just because

rh=$(grep -Po "Closest.*radius.*:\K.*" uammd.pairMobility.log | awk '{print $1}')

gfortran -O3 tools/ewaldSumRPY.f90 -o ewaldSumRPY; echo "$rh $L $L $L" | ./ewaldSumRPY | awk '{print $1/'$hydrodynamicRadius', $2}' > theo; rm ewaldSumRPY

gracebat theo pairMobility_pullForce.test -par tools/pairMobility.par -hdevice EPS -printfile pairMobility.eps
rm theo

mv selfMobility_pullForce*.test pairMobility_pullForce*.test uammd*log $resultsFolder/

./fib selfDiffusion $temperature $viscosity $hydrodynamicRadius  $scheme > uammd.selfDiffusion.log 2>&1 
g++ -O3 -std=c++11 tools/msd.cpp tools/Timer.cpp -o msd


rh=$(grep -Po "Closest.*radius.*:\K.*" uammd.selfDiffusion.log | awk '{print $1*1}')
dt=$(grep -Po 'dt:\K.*$' uammd.selfDiffusion.log | awk '{print $1*1}')
file=$(ls pos.noise.dt*.test)

for i in $(ls pos.noise.*.test)
do
    dt=$(echo $i | grep -Po '\.dt\K.*(?=.L)')
    L=$(echo $i | grep -Po '\.L\K.*(?=.M)')
    M0=$(echo $i | grep -Po '\.M\K.*(?=.test)')
./msd -N 4096 -Nsteps 1500 $i 2>/dev/null |
    awk '{MSD0=(2*$1*'$dt'*'$M0'*'$temperature');
	      print $1*'$dt', $2/MSD0, $3/MSD0, $4/MSD0}' |
    head -10 | #Compute mean and std
    tee $i.msd |
    awk '{c++; 
	      d=$2-mx; mx += d/c; d2 = $2-mx; x2 += d*d2;
	      d=$3-my; my += d/c; d2 = $3-my; y2 += d*d2;
	      d=$4-mz; mz += d/c; d2 = $4-mz; z2 += d*d2;
	     }
	     END{
	     print mx, my, mz, x2/c, y2/c, z2/c,'$L';
 	     }'	 > $i.diffusion
done
rm msd
awk '{printf "%e %e %e \n", $7, $1, $4 >> "diff.x"; 
      printf "%e %e %e \n", $7, $2, $5 >> "diff.y";
      printf "%e %e %e \n", $7, $3, $6 >> "diff.z";}' pos.noise.*diffusion

gracebat -type xydy diff.x diff.y diff.z -par tools/selfDiffusion.par -hdevice EPS -printfile selfDiffusion.eps


paste diff.{x,y,z} | awk '{print $1, $2, $3, $5,$6, $8, $9}' > $resultsFolder/selfDiffusion.dat
mv diff.{x,y,z} $resultsFolder/
mv pos.noise* $resultsFolder/


./fib noiseVariance $temperature $viscosity $hydrodynamicRadius $scheme > uammd.noiseVariance.log 2>&1 
cat noiseVariance.test |
    awk '{var=2; print $1, $2/var, $3/var, $4/var}' > kk


gracebat -nxy kk -par tools/noiseVariance.par -hdevice EPS -printfile noiseVariance.eps

rm kk
mv noiseVariance.test $resultsFolder/


mkdir -p $figuresFolder

mv *.eps $figuresFolder
mv uammd.*log $resultsFolder/



