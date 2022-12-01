#Raul P. Pelaez 2022. Compute wall impedance from the velocity field v_x(z)
datamain=$1

freq=$(awk '/^wallFreq/{print $2}' $datamain)
w=$(echo $freq | awk '{print 2*3.141592653589793*$1}')
v0=$(awk '/^wallAmplitude/{print $2}' $datamain)
vis=$(awk '/^shearViscosity/{print $2}' $datamain)
rho=$(awk '/^initialDensity/{print $2}' $datamain)
L=$(awk '/^boxSize/{print $4}' $datamain)
nz=$(awk '/^cellDim/{print $4}' $datamain)
h=$(echo $L $nz | awk '{print $1/$2}')
delta=$(echo 1 | awk '{print sqrt(2*'$vis'/('$rho'*'$w'))}')
#echo "Delta is: $delta" > /dev/stderr

simulationTime=$(awk '/^simulationTime/{print $2}' $datamain)
period=2000
firstTime=0
for t in $(seq $firstTime $period $simulationTime)
do
    if ! test -f vel.uammd.$t; then  continue; fi
    v1=$(head -1 vel.uammd.$t | awk '{print $2}')
    imp=$(echo $v1 | awk '{print 2.0/'$h'*($1/'$v0' - cos('$w'*'$t'))*'$delta'}')
    imp_theory=$(./impedance $vis $L $v0 $t $w | awk '{print $1}')
    echo $t $imp $imp_theory $(echo 1 | awk '{print '$imp'-('$imp_theory')}')
done >imp.real

firstTime=500
for t in $(seq $firstTime $period $simulationTime)
do
    if ! test -f vel.uammd.$t; then  continue; fi
    v1=$(head -1 vel.uammd.$t | awk '{print $2}')
    imp=$(echo $v1 | awk '{print 2.0/'$h'*($1/'$v0' - cos('$w'*'$t'))*'$delta'}')
    imp_theory=$(./impedance $vis $L $v0 $t $w | awk '{print $2}')
    echo $t $imp $imp_theory $(echo 1 | awk '{print '$imp'-('$imp_theory')}')
done > imp.imag
paste imp.{real,imag} | awk '{print $1, $4,$8}' > imp.error
