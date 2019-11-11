#Compute potential between two charges at several distances

#Computes the potential for several box sizes and extrapolates to L->inf
tolerance=1e-7
gw=1;

rmin=2
rmax=24
dr=2

lmin=64
lmax=350
dl=8

make poisson

for r in $(seq $rmin $dr $rmax | sort -R);
do
    for l in $(seq $lmin $dl $lmax | sort -R);
    do
	./poisson $l $r $tolerance 2> log |
	    awk '{print $7}' |
	    tr '\n' ' ' |
	    awk '{print '$l', '$r', $0}';
	
    done > pot.r$r ;
done

rm log;

outfile=coulomb_vs_r_deviation.dat
for r in $(seq $rmin $dr $rmax);
do
    a=$(gnuplot -e 'f(x) = a+b/x+b/x**2+c/x**3+d/x**4+e/x**5; 
	 set fit quiet;
	 fit f(x) "pot.r'$r'" u 1:3 via a,b,c,d,e;
	 gw = '$gw';
	 r = '$r';
	 theo = 1/(4*gw*pi**1.5)-1/(4*pi*r)*erf(r/(2*gw));
	 deviation = abs(1.0-a/theo);

	 set print "-"; print("%.15g", deviation);');
    echo $r $a
done > $outfile

echo "Maximum deviation from theory:"
cat $outfile | awk '$2>max{max=$2;maxr=$1} END{print "distance:", maxr, " | deviation:", max}'

rm -f fit.log
