set -e

#Compile and run UAMMD test
make
echo "Running UAMMD test"
./bvp 2>&1 |
    awk '!/\[/{for(i=1; i<=NF; i++){printf("%25s ", $i);} printf("\n");next}1' |
    tee uammd.out
cat uammd.out |
    awk '/PRINTING/{p=1;next}p&&($1!="z"){print $0}' |
    awk 'NF{print  $2 > "fn.dat"; print $4 > "yn.uammd" }'
#Run matlab test
echo "Running matlab test"
octave -q bvp.m |
    awk '/OUTPUT BEGIN/{p=1;next}p&&NF{print $1}' > yn.matlab

echo "Printing comparison"
#Compare both results, use maximum value for normalizing error
max=$(cat yn.matlab | awk 'sqrt($1*$1*1)>a{a=sqrt($1*$1)}END{print a}')
printf "%25s %25s %25s\n" MATLAB UAMMD ERROR
paste yn.matlab yn.uammd |
    awk '{print $1, $2, (($1)-($2))/'$max'}' |
    awk '{for(i=1; i<=NF; i++){printf("%25s ", $i);} printf("\n");}' |#Just formatting
    tee error.dat

echo "Maximum error is: " $(cat error.dat | awk '{print sqrt($3*$3)}' | awk '$1>a{a=$1}END{print a}')
