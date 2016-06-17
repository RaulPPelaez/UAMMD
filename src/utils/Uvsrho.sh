


make -j9 > /dev/null

dmin=0.1
dmax=1.6
Nd=50

for i in {0..50}
do
    d=$(echo "($dmax-$dmin)*$i/$Nd+$dmin" | bc -l)
    
    ./main $d
    
    u=$(awk '{s+=$1} END { print s/NR;}' measurables.dat)
    t=$(awk '{s+=$2} END { print 2.0*s/NR/3.0;}' measurables.dat)
    p=$(awk '{s+=$4} END { print s/NR;}' measurables.dat)
    echo $d $t $u $p

done
