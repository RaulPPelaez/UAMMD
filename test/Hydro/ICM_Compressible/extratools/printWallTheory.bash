
datamain=$1
freq=$(awk '/^wallFreq/{print $2}' $datamain)
v0=$(awk '/^wallAmplitude/{print $2}' $datamain)
vis=$(awk '/^shearViscosity/{print $2}' $datamain)
rho=$(awk '/^initialDensity/{print $2}' $datamain)
L=$(awk '/^boxSize/{print $4}' $datamain)
nz=$(awk '/^cellDim/{print $4}' $datamain)
h=$(echo $L $nz | awk '{print $1/$2}')

delta=$(echo 1 | awk '{print sqrt(2*'$vis'/('$rho'*'$w'))}')
printTime=$(awk '/^printTime/{print $2}' $datamain)
simulationTime=$(awk '/^simulationTime/{print $2}' $datamain)

time=1000
for z in $(cat vel.dat | awk '/#/{p++;next}p==2{exit}{print $1}')
do
theo=$(../walltheory $vis $L $v0 $time $freq $z)
echo $z $theo
done > wall.theo.$time
