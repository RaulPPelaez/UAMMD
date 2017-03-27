gfortran -O3 lj_eq_state.f 
#rm -rf UvsrhoT1.dat.teo

rm -f test.dat

#while read d t u p
##for i in {3..10} 
#do
##    t=1.1 #$(echo "0.2*$i" | bc -l)
##    for j in {1..100}
##    do
#    t2=$(echo "$t*2" | bc -l)
#	echo $d $t >/tmp/kk
#	./a.out < /tmp/kk >> test.dat
##    done
##    echo " " >> test.dat
##done
#done < ../../DTUP4.dat
#while read d t u p
d=0.2
for t in 0.4 1.0 2.0
do
#    t=$(echo "2+0.01*$i" | bc -l)
#    for j in {1..100}
#    do
	echo $d $t >/tmp/kk
	./a.out < /tmp/kk # >> test.dat
done
#    echo " " >> test.dat
#done
#done < ../../DTUP4.dat

