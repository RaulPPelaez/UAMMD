H=$(grep -oP "H\K.*" data.main | awk '{print $1*1.0}')
L=$(grep -oP "Lxy\K.*" data.main | awk '{print $1*1.0}')
seed=$(head -100 /dev/urandom | cksum | awk '{print $1}')
N=$(grep -oP "numberParticles\K.*" data.main | awk '{print $1*1.0}')
seq $N |
    awk 'BEGIN{srand('$seed')}
    	{acc=0;
    	 while(acc==0){
		H='$H';
		x=(rand()-0.5)*2;
		y=(rand()-0.5)*2;
		z=(rand()-0.5)*(1.0-2.0/H);
		p=1;
		Z=rand();
		if(Z<p){
		    acc=1;
		    print x,y,z;
		}
	  }
	}' |
    awk '{printf "%.14g %.14g %.14g %g\n", $1*'$L'*0.5, $2*'$L'*0.5, $3*'$H', (NR%2-0.5)*2;}'
