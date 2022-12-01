K=$1
datamain="data.main"
H=$(grep -Eo "^H[[:space:]].*" $datamain | awk '{print $2}')
L=$(grep -Eo "^Lxy[[:space:]].*" $datamain | awk '{print $2}')
numberParticles=$(grep -Eo '^numberParticles[[:space:]].*' $datamain | awk '{print $2}')
seed=$(head -100 /dev/urandom | cksum | awk '{print $1}')
seq $numberParticles |
    awk 'BEGIN{srand('$seed')}
        {acc=0;
	 K='$K'/(H - 2.0);
	 M=1/cos(K*0.5*H)^2;
         while(acc==0){
                H='$H';
                x=(rand()-0.5)*'$L';
                y=(rand()-0.5)*'$L';
                z=(rand()-0.5)*(H-2.0);
                p=(1.0/(cos(K*z))^2)/M;
                Z=rand();
                if(Z<p){
                    acc=1;
                    print x,y,z,1;
                }
          }
        }'
