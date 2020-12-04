positions=$1
datamain="data.main"
nbins=200
Hhalf=$(grep -Eo '^H[[:space:]].*' $datamain | awk '{print $2*0.5}')
nsteps=$(cat $positions | grep -c "#");
N=$(grep -Eo '^numberParticles[[:space:]].*' $datamain | awk '{print $2}')

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
    awk '{printf "%.14g %.14g\n", $3,('$nbins'/('$N'*'$nsteps'))}' |
    histogram $nbins $Hhalf -$Hhalf
