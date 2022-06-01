#Raul P. Pelaez 2022. Test script for the compressible ICM.
# Runs test.cu to compute dynamic structure factors for a number of identical runs. Then averages the results.
#Prints the files S*.dat, containing several cross correlations.
# The parameters for the run are in data.main
set -e
rm -rf run*
make -B

nruns=10 #Number of identical runs to average
ndev=1 #Number of available GPUs


dev=0
for i in $(seq $nruns)
do
    if [ $dev  -eq $ndev ]
    then
        dev=0
	wait
    fi
    (
	mkdir -p run$i
	cd run$i
	../test ../data.main --device $dev 2> log
    )&
    dev=$(echo $dev | awk '{print $1+1}')
    echo "Run $i"
done

wait

function averageRuns(){
    name=$1
    cat run*/S${name}qw*.dat |
    awk '{print $2,$3}' |
    datamash --sort -W groupby 1 mean 2 sstdev 2 |
    sort -g > S${name}.dat
}

averageRuns vxvy
averageRuns vxvx
averageRuns rhovx
averageRuns rhorho
