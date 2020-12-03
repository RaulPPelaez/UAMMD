
tmp=$(mktemp)
folder=$1
for r in $(ls -d $folder)
do
    referenceN=$(ls $r/force* |  sed -r -e 's+.*force([0-9]\+).*+\1+g' | sort -g -k1 | sed -n ''$(ls $r/force*dat | wc -l | awk '{print int($1/2)}')'p' )
    reference="$r/force$referenceN.dat"
    samples=$(cat $reference | wc -l)
    sumRef=$(cat $reference | awk '{printf "%.14g %.14g %.14g\n", $2*$2, $3*$3, $4*$4}' | datamash -W sum 1-3)
    sX=$(echo $sumRef | awk '{print sqrt($1/'$samples')}')
    sY=$(echo $sumRef | awk '{print sqrt($2/'$samples')}')
    sZ=$(echo $sumRef | awk '{print sqrt($3/'$samples')}')


    first="$r/force$(ls $r/force*|  sed -r -e 's+.*force([0-9]\+).*+\1+g' | sort -g -k1 | head -1 ).dat"
paste $first $reference |
    awk '{printf "%.14g %.14g %.14g\n", ($2-$6)/'$sX', (($3-$7))/'$sY', ($4-$8)/'$sZ'}'
done  > $tmp
Nh=200
N=18920
min=$(cat $tmp | datamash -W min 1)
max=$(cat $tmp | datamash -W max 1)
nsam=$(ls -d $folder | wc -l | awk '{print $1*'$N'*1.0}')
cat $tmp |
    awk '{print $1, 1}' |
    awk '{min='$min'; max='$max'; Nh='$Nh';b=int(($1-min)/(max-min)*Nh); h[b] += $2;}
                 	 END{for(i=0;i<Nh;i++){z=(i+0.5)/Nh*(max-min)+min; print z, h[i]*1;}}' | awk '{print $1, $2/'$nsam'}' > histX.dat
min=$(cat $tmp | datamash -W min 2)
max=$(cat $tmp | datamash -W max 2)
cat $tmp |
    awk '{print $2, 1}' |
    awk '{min='$min'; max='$max'; Nh='$Nh';b=int(($1-min)/(max-min)*Nh); h[b] += $2;}
                 	 END{for(i=0;i<Nh;i++){z=(i+0.5)/Nh*(max-min)+min; print z, h[i]*1;}}' | awk '{print $1, $2/'$nsam'}' > histY.dat
min=$(cat $tmp | datamash -W min 3)
max=$(cat $tmp | datamash -W max 3)
cat $tmp |
    awk '{print $3, 1}' |
    awk '{min='$min'; max='$max'; Nh='$Nh';b=int(($1-min)/(max-min)*Nh); h[b] += $2;}
                 	 END{for(i=0;i<Nh;i++){z=(i+0.5)/Nh*(max-min)+min; print z, h[i]*1;}}' | awk '{print $1, $2/'$nsam'}' > histZ.dat

rm $tmp
