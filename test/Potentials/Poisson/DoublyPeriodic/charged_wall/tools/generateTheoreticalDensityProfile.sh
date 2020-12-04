datamain="data.main"
K=$1
H=$(grep -Eo '^H[[:space:]].*' $datamain | awk '{print $2}')
nbins=100
N=1e7

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

seq $N | awk '{H='$H';
	        z=(rand())*(H*0.5);
		Kd='$K'/(H-2.0);
		theo=(z<(H*0.5-1)&&z>-(H*0.5-1))/cos(Kd*z)**2;
		print z, theo*('$nbins'/'$N');
	       }' | histogram $nbins $(echo $H | awk '{print $1*0.5;}') 0

