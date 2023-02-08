#Raul P. Pelaez 2022. Test script for the compressible ICM.
#Measures the RDF of a LJ liquid. Compares with an equivalent BD simulation.
set -e

folder=rdf_simulations
ndev=$1 #Number of available GPUs
nruns=5 #Number of identical runs to average
naverage=1 #Number of times to run $nruns for statistics

datamain=data.main.rdf

function downloadRDF(){
    mkdir -p tools
    (
	cd tools
	if ! test -f rdf
	then
	    git clone https://github.com/raulppelaez/RadialDistributionFunction
	    cd RadialDistributionFunction
	    mkdir build && cd build && cmake .. && make -j4; cd ..
	    cp build/bin/rdf ..
	    cd ..
	    rm -rf RadialDistributionFunction
	fi
    )
}

function average(){
    local runs=$1
    local name=$3
    local outputfolder=$2
    cat ${runs}*/${name} |
    datamash --sort -W groupby 1 mean 2 sstdev 2 |
    sort -g > $outputfolder/${name}
}

function generateTheory(){
    mkdir -p $folder/theory
    (
	cd $folder/theory
	cat ../../$datamain | awk '/^scheme/{print "scheme bd";next}1' > $datamain.theory
	../../particles $datamain.theory 2> log.theory > theory.pos
	local nsteps=$(grep -c "#" theory.pos)
	local numberParticles=$(awk '/^numberParticles/{print $2}' $datamain.theory)
	local lx=$(awk '/^boxSize /{print $2}' $datamain.theory)
	local maxDist=$(echo $lx | awk '{print $1*0.125}')
	cat theory.pos | ../../tools/rdf -N $numberParticles -Nsnapshots $nsteps -L $lx -nbins 500 -rcut $maxDist  > theory.rdf
    )
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
	    local nsteps=$(grep -c "#" particles.pos)
	    local numberParticles=$(awk '/^numberParticles/{print $2}' $datamain)
	    lx=$(awk '/^boxSize /{print $2}' $datamain)
	    maxDist=$(echo $lx | awk '{print $1*0.125}')
	    cat particles.pos | ../../../tools/rdf -N $numberParticles -Nsnapshots $nsteps -L $lx -nbins 500 -rcut $maxDist  > particles.rdf
	)&
	dev=$(echo $dev | awk '{print $1+1}')
	printf " $i..."
    done
    wait
    printf "DONE\n"
    average $subfolder/run $subfolder particles.rdf
}

make particles
mkdir -p $folder

downloadRDF
generateTheory

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
    average $folder/runAvg $folder particles.rdf
done


mkdir -p figures
gracebat $folder/particles.rdf $folder/theory/theory.rdf -par tools/rdf.par -hdevice EPS -hardcopy -printfile figures/rdf.eps
