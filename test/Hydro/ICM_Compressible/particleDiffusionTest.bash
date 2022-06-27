#Raul P. Pelaez 2022. Test script for the compressible ICM.
# Places a group of particle tracesrs in the fluid and lets them evolve.
# Then measures their mean square displacement and compares with the expected long time value, D0.
# Generates a figure with the diffusion coefficient in each direction. This figure should show three equivalent lines converging to 1.
# Runs the particleDiffusionTest.cu code.
set -e

folder=diffusion_simulations
ndev=$1 #Number of available GPUs
nruns=5 #Number of identical runs to average
naverage=3 #Number of times to run $nruns for statistics

datamain=data.main.diffusion

function downloadMSD(){
    mkdir -p tools
    (
	cd tools
	if ! test -f msd
	then
	    git clone https://github.com/raulppelaez/MeanSquareDisplacement
	    cd MeanSquareDisplacement
	    mkdir build && cd build && cmake .. && make -j4; cd ..
	    cp build/bin/msd ..
	    cd ..
	    rm -rf MeanSquareDisplacement
	fi
    )
}

function average(){
    local runs=$1
    local name=$3
    local outputfolder=$2
    cat ${runs}*/${name} |
    datamash --sort -W groupby 1 mean 2-4 sstdev 2-4 |
    sort -g > $outputfolder/${name}
}

function runSimulations(){
    local nruns=$1 #Number of identical runs to average
    local subfolder=$folder/$2
    local dev=0
    printf "Running... "
    for i in $(seq $nruns)
    do
	if [ $dev  -eq $ndev ]
	then
            dev=0
	    wait
	fi
	(
	    mkdir -p $subfolder/run$i
	    cp $datamain $subfolder/run$i
	    cd $subfolder/run$i
	    ../../../particles $datamain --device $dev 2> log > particles.pos
	    nsteps=$(grep -c "#" particles.pos)
	    numberParticles=$(awk '/^numberParticles/{print $2}' $datamain)
	    cat particles.pos | ../../../tools/msd -N $numberParticles -Nsteps $nsteps > particles.msd
	)&
	dev=$(echo $dev | awk '{print $1+1}')
	printf " $i..."
    done
    wait
    printf "DONE\n"
    average $subfolder/run $subfolder particles.msd
}

make particles
mkdir -p $folder
downloadMSD
for i in $(seq $naverage)
do
    subfolder=runAvg$i
    if test -d $folder/$subfolder
    then
	continue
    fi
    echo "Average run number $i"
    mkdir -p $folder/$subfolder
    runSimulations $nruns $subfolder
    average $folder/runAvg $folder particles.msd
done


dt=$(awk '/^printTime /{print $2}' $datamain)
viscosity=$(awk '/^shearViscosity /{print $2}' $datamain)
T=$(awk '/^temperature /{print $2}' $datamain)
lx=$(awk '/^boxSize /{print $2}' $datamain)
nx=$(awk '/^cellDim /{print $2}' $datamain)
a=$(echo 1 | awk '{print 0.91*'$lx'/'$nx'}')
D0=$(echo 1 | awk '{print '$T'/(6*3.1415*'$viscosity'*'$a')}')
cat $folder/particles.msd | awk '{printf("%g ", $1*'$dt'); for(i=2;i<=7; i++) printf("%g ",$i/(2*'$D0')); printf("\n")}' > $folder/particles.msd.norm

awk '$1{print $1, $2/$1, $5/$1}' particles.msd.norm > X
awk '$1{print $1, $3/$1, $6/$1}' particles.msd.norm > Y
awk '$1{print $1, $4/$1, $7/$1}' particles.msd.norm > Z
mkdir -p figures
gracebat -type xydy X Y Z -par tools/particleDiffusion.par -hdevice EPS -hardcopy -printfile figures/msd.eps
rm -f X Y Z
