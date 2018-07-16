gfortran -O3 lj_eq_state.f 

d=$1
t=$2
rc=$3
sigma=$4
epsilon=$5
tmp=$(mktemp)

echo $d $t $rc $sigma $epsilon >$tmp
./a.out < $tmp 

rm $tmp a.out

