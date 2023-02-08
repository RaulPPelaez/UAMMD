#Raul P. Pelaez 2022. Moving wall test.
#Compares UAMMD results with theory for a domain walled in z. The bottom wall moves with an oscillatoy velocity


folder=walltest_data
datamain=data.main.wall

mkdir -p $folder
make walltest

g++ -std=c++14 tools/walltheory.cpp -o walltheory

viscosity=$(awk '/^shearViscosity/{print $2}' $datamain)
Lz=$(awk '/^boxSize/{print $4}' $datamain)
vamplitude=1
printTime=$(awk '/^printTime/{print $2}' $datamain)
simulationTime=$(awk '/^simulationTime/{print $2}' $datamain)


(
    cd $folder
    ../walltest ../$datamain 2> log
    cat vel.dat | sed 's+#+ +g' > vel.uammd
)

for i in $(seq 0 $printTime $simulationTime); do
    for z in $(cat vel.dat | awk '/#/{p++;next}p==2{exit}{print $1}')
    do
	theo=$(./walltheory $vis $L $v0 $time $freq $z)
	echo $z $theo
    done
    echo " ";
done > $folder/vel.theory

mkdir -p figures
gracebat $folder/vel.uammd $folder/vel.theory -par tools/walltest.par -hdevice EPS -hardcopy -printfile figures/walltest.eps
