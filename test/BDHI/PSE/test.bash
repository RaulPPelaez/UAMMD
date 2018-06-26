


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

make clean pse

mkdir -p results

./pse > uammd.log 2>&1 

gracebat selfMobility_pullForce.psi*.test -par tools/selfMobility.par -hdevice EPS -printfile selfMobility.eps

gfortran -O3 tools/ewaldSumRPY.f90 -o ewaldSumRPY; ./ewaldSumRPY > theo; rm ewaldSumRPY

gracebat theo pairMobility_pullForce.psi*.test -par tools/pairMobility.par -hdevice EPS -printfile pairMobility.eps
rm theo

mv selfMobility_pullForce*.test pairMobility_pullForce*.test results/


g++ -O3 -std=c++11 tools/msd.cpp tools/Timer.cpp -o msd
for i in $(ls pos.noise*.test)
do
    dt=$(echo $i | awk -F dt '{print $2}' | awk -F .test '{print $1}')
    ./msd -N 4096 -Nsteps 2000 $i 2>/dev/null |
	awk '{M0=(2*$1*'$dt'/(6*3.1415)*(1-2.837/64));
	      print $1*'$dt', $2/M0, $3/M0, $4/M0}' > $i.msd;
done

rm msd
for i in $(ls pos.noise.psi*.test.msd);
do
    head -50 $i | #Compute mean and std
	awk '{c++; 
	      d=$2-mx; mx += d/c; d2 = $2-mx; x2 += d*d2;
	      d=$3-my; my += d/c; d2 = $3-my; y2 += d*d2;
	      d=$4-mz; mz += d/c; d2 = $4-mz; z2 += d*d2;
	     }
	     END{
	     print mx, my, mz, x2/c, y2/c, z2/c;
 	     }'	 > $i.diffusion;
done


for i in $(ls *.diff* | cut -d. -f3,4 | sed 's+psi++')
do
    awk '{printf "%e %e %e \n", '$i', $1, $4 >> "diff.x"; 
    	  printf "%e %e %e \n", '$i', $2, $5 >> "diff.y";
	  printf "%e %e %e \n", '$i', $3, $6 >> "diff.z";}' pos.noise.psi$i.*diffusion
done


gracebat -type xydy diff.x diff.y diff.z -par tools/selfDiffusion.par -hdevice EPS -printfile selfDiffusion.eps

rm diff.{x,y,z}
rm pos.noise*diffusion
mv pos.noise* results/


cat noiseVariance.test |
    awk '{M0 = $5; var=2*M0; print $1, $2/var, $3/var, $4/var}' > kk


gracebat -nxy kk -par tools/noiseVariance.par -hdevice EPS -printfile noieVariance.eps

rm kk
mv noiseVariance.test results/


mkdir -p figures

mv *.eps figures
mv uammd.log results/
