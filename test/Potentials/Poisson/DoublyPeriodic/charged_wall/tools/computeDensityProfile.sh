positions=$1
datamain="data.main"
K=$2
nbins=100

nsteps=$(cat $positions | grep -c "#");
Hhalf=$(grep -Eo '^H[[:space:]].*' $datamain | awk '{print $2*0.5}')
N=$(grep -Eo '^numberParticles[[:space:]].*' $datamain | awk '{print $2}')
ep=$(grep -Eo '^permitivity[[:space:]].*' $datamain | awk '{print $2}')

nm=$(echo 1 | awk '{printf "%.14g\n", 2*'$K'**2*'$ep'/(1.0-1.0/'$Hhalf')**2;}') 

function histogram {
nbins=$1
upper=$2
lower=$3
awk '{min='$lower'; max='$upper'; nbins='$nbins';b=int(($1-min)/(max-min)*nbins); h[b] += $2;}
     END{for(i=0;i<nbins;i++){
	   z=(i+0.5)/nbins*(max-min)+min;
	   print z, h[i]*1.0;
	 }
	}'
}

cat $positions |
    grep -v "#" |
    awk '{printf "%.14g %.14g\n", sqrt($3*$3),('$nbins'/('$nm'*'$N'*'$nsteps'))}' |
    histogram $nbins $Hhalf 0
