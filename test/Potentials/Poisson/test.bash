#Compute potential between two charges at several distances

#Computes the potential for several box sizes and extrapolates to L->inf
tolerance=1e-7
gw=0.001;
split=0.1

rmin=2
rmax=24
dr=4

lmin=64
lmax=350
dl=16

make poisson
outfile=electric_field_vs_r_deviation.dat
for r in $(seq $rmin $dr $rmax);
do
    for l in $(seq $lmin $dl $lmax);
    do
	./poisson $l $r $tolerance $gw $split 2> log |
	    awk '{print $7}' |
	    tr '\n' ' ' |
	    awk '{print '$l', '$r', $0}';
	
    done > field.r$r ;
    a=$(gnuplot -e 'f(x) = a+b/x+b/x**2+c/x**3+d/x**4+e/x**5; 
	 fit f(x) "field.r'$r'" u 1:3 via a,b,c,d,e;
	 set fit quiet;
	 gw = '$gw';
	 r = '$r';
	 theo = -(exp(-r**2/(4.0*gw**2))/(4*pi**1.5*gw*r) - erf(r/(2.0*gw))/(4*pi*r**2));
	 deviation = abs(1.0-a/theo);
	 set print "-"; print("%.15g", deviation);');
    echo $r $a

done > $outfile

rm log;

echo "Maximum deviation from theory:"
cat $outfile | awk '$2>max{max=$2;maxr=$1} END{print "distance:", maxr, " | deviation:", max}'

rm -f fit.log
