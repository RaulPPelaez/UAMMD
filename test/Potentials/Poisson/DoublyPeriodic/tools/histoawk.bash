H=$(grep -oP "H\K.*" data.main | awk '{print $1*1.0}')
Nh=$1
awk '{min=-'$H'*0.5; max= '$H'*0.5; Nh='$Nh';b=int(($1-min)/(max-min)*Nh); h[b] += $2;}
           	 END{for(i=0;i<Nh;i++){
		 	     z=(i+0.5)/Nh*(max-min)+min;
	     print z, h[i];
	     }}' 
