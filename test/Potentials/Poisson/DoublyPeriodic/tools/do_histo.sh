#file=$1
#steps=$2
#Nbins=$3
#N=$(ifile $file | awk '{print $2}')
#Nlines=$(echo $steps $N | awk '{print $1*($2+1)}')
#tmppos=$(mktemp)
#tail -$Nlines $file | awk '!/#/{print $3}' > $tmppos;
#H=$(grep -oP "H\K.*" data.main)
#L=$(grep -oP "Lxy\K.*" data.main)
#octave --eval "H=$H;L=$L;Nh=$Nbins; sigma=1.0*$N/(L^2); a=load('$tmppos');[h,x]=hist(a,linspace(-H*0.5, H*0.5,Nh)); h = h/$steps*Nh/H;format long; disp([x' h'])"
#rm $tmppos


#K=2.975
#N=20711

K=2.65
N=6140

file=$1
cat $file |
    awk '!/#/{print $3, 1}' |
    bash histoawk.bash  |
    awk '{K='$K'/49.0; print $1, $2*'$N'/50.0/400**2/(2*K*K*4.438411913494056e-02)}'
