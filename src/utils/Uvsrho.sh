

T=1
make -j9

dmin=0.01
dmax=0.8
Nd=50

for i in {0..50}
do
    d=$(echo "($dmax-$dmin)*$i/$Nd+$dmin" | bc -l)
    
    ./main $d $T
    
    u=$(awk '{s+=$1} END { print s/NR;}' measurables.dat)
    t=$(awk '{s+=$2} END { print s/NR/3.0;}' measurables.dat)
    p=$(awk '{s+=$4} END { print s/NR;}' measurables.dat)
    echo $d $t $u $p

done
