#Raul P. Pelaez 2022. Test script for the compressible ICM.
# Runs test.cu to compute dynamic structure factors for a number of identical runs. Then averages the results.
#Prints the files S*.dat, containing several cross correlations.
# Also plots the results, comparing to theory.
# The parameters for the run are in data.main. But theory is only available for the default parameters.
set -e
make structureFactorTest
folder=structure_simulations

mkdir -p $folder

ndev=$1 #Number of available GPUs
nruns=20 #Number of identical runs to average
naverage=3 #Number of times to run $nruns for statistics

function average(){
    local runs=$1
    local name=$3
    local outputfolder=$2
    cat ${runs}*/${name}.dat |
    datamash --sort -W groupby 1 mean 2 sstdev 2 |
    sort -g > $outputfolder/${name}.dat
}


function correlate(){
    local x=${1}
    local y=${2}
    local location=$3
    local k=$4
    local V=$(grep "^boxSize" data.main | awk '{print $2*$3*$4}')
    local nsteps=$(cat $location/log | grep "Number of steps" | awk '{print $5}')
    local T=$(grep "^dt" data.main | awk '{print $2*'$nsteps'}')
    local norm=$(echo $V $T | awk '{print $1*$2}')
    #local dV=$(grep "^cellDim" data.main | awk '{print '$V'/($2*$3*$4)}')
    paste $location/${x}qw${k}.dat $location/${y}qw${k}.dat | awk '{xr = $2; xi = $3; yr =$5; yi=$6; print $1, '$norm'*(xr*yr + xi*yi), '$norm'*(xi*yr - xr*yi);}' |
	datamash --sort -W groupby 1 mean 2,3 |
	sort -g > $location/S$x$y.dat
}

function correlateAll(){
    local outputFolder=$1
    local waveNumber=994
    correlate rho rho $outputFolder $waveNumber
    correlate vx vx $outputFolder   $waveNumber
    correlate vx vy $outputFolder   $waveNumber
    correlate rho vx $outputFolder  $waveNumber
}

function averageAll(){
    local inputFolder=$1
    local outputFolder=$2
    average $inputFolder $outputFolder Srhorho
    average $inputFolder $outputFolder Svxvx
    average $inputFolder $outputFolder Svxvy
    average $inputFolder $outputFolder Srhovx
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
	    cp data.main $subfolder/run$i
	    cd $subfolder/run$i
	    ../../../structureFactorTest data.main --device $dev 2> log
	    correlateAll .
	)&
	dev=$(echo $dev | awk '{print $1+1}')
	printf " $i..."
    done
    wait
    printf "DONE\n"
    averageAll $subfolder/run $subfolder
}


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
    averageAll $folder/runAvg $folder
done

function savePlot(){
    local name=$1
    mkdir -p figures
    gracebat  tools/theory/$name.dat $folder/$name.dat -par tools/$name.par -hdevice EPS -hardcopy -printfile figures/$name.eps
}
savePlot Srhorho
savePlot Srhovx
savePlot Svxvx
savePlot Svxvy
