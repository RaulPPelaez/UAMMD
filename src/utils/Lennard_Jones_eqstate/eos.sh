gfortran -O3 lj_eq_state.f 
rm -rf UvsrhoT1.dat.teo


while read d t u p
do
echo $d $t >/tmp/kk
./a.out < /tmp/kk >>UvsrhoT1.dat.teo
done < ../../UvsrhoT1.dat

