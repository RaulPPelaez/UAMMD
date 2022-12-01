
ndev=$1
folder=impedance_simulations
mkdir -p $folder
lxy=16
lz=64
tools=extratools
g++ $tools/impedance.cpp -o impedance
g++ $tools/walltheory.cpp -o walltheory
dev=0
for h in 0.125 0.25 0.5 1 2
do
    subfolder=run_h$h
    if test -d $folder/$subfolder
    then
	continue
    fi
    mkdir -p $folder/$subfolder
    datamain=data.main.wall.$h
    nxy=$(echo $lxy | awk '{print int($1/'$h');}')
    nz=$(echo $lz | awk '{print int($1/'$h');}')
    cat<<EOF>$folder/$subfolder/$datamain
boxSize $lxy $lxy $lz
cellDim $nxy $nxy $nz
wallAmplitude 3.141592653589793e-03
shearViscosity 0.226194671058465
bulkViscosity 0
initialDensity 1
speedOfSound 2.0
temperature 0
dt 0.025
#Other scripts are hardcoded to a period of time=2000, they will break if you change this
wallFreq 0.0005
relaxTime 20000
printTime 500
simulationTime 20000
EOF
    if [ $dev  -eq $ndev ]
    then
        dev=0
	wait
    fi
    (cd $folder/$subfolder
     ../../walltest $datamain --device $dev 2> log
     cp ../../impedance .
     bash ../../$tools/splitVeldat.bash $datamain
     bash ../../$tools/impedance.bash $datamain
    )&
    dev=$(echo $dev | awk '{print $1+1}')
    printf " h=$h..."
done
wait
echo "DONE!"
