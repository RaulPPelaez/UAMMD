tar xvf init.tar.xz

dt=0.1
wc=1.0
for T in $(seq 0.2 0.01 1.2 | shuf)
do
    echo "Doing $T"
    sed -i 's+^temperature+temperature '$T'+' data.main
    ./deserno 2> uammd.T$T.log
    echo "Simulation finished"
    S=$(cat pos.dat |
	awk 'NF==1{if(St)print St/Sc;St=0; Sc=0;next}
    	!$5{St+=0.5*S;Sc=Sc+1; ax=$1; ay=$2; az=$3}
	$5{S= 3*(($3-az)/sqrt(($1-ax)**2+($2-ay)**2+($3-az)**2))**2-1;}' |
    tail -100 |
    datamash mean 1 sstdev 1)
    name=pos.T$T.wc$wc.dat
    mv pos.dat $name
    echo "Computing msd"
    tmp=$(mktemp)
    cat $name | awk 'NF==1||!$5{print $1, $2, $3}' > $tmp
    msd -N 4608 -Nsteps 999 $tmp 2>/dev/null | tail -50 > $name.msd
    ab=$(gnuplot -e 'f(x) = a*x+b; fit f(x) "'$name.msd'" u 1:(($2+$3)*0.5) via a,b' 2>&1  |
	     grep = |
	     grep '^a\|^b' |
	     awk '{print $3}' |
	     paste -sd" " | awk '{print $1/(2*'$T'*'$dt'/(6*3.1415)), $2;}')	
    rm -f $tmp fit.log

    echo $T $S $ab | tee -a deserno.test
    
done
