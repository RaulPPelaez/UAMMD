
d=$1
t=$2

g++ -O3 lj_eos.cpp -o lj_eos 2> /dev/null

awk 'BEGIN{print '$t'; print '$d';}' |
    ./lj_eos |
    grep 'In\|Pre' |
    awk -F ":" '{print $2}' | awk '{print $1}' |
    tr '\n' ' ' | 
    awk '{print '$d', '$t', $2+1.5*'$t', $1}'


