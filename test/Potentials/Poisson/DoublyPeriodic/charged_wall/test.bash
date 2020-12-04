
if ! test -f data.main; then echo "ERROR: This script expects a data.main to be present" >/dev/stderr; exit 1; fi
if ! test -f ../poisson; then echo "ERROR: This script poisson to be compiled and available at ../ " >/dev/stderr; exit 1; fi

K=2.65 #This K value corresponds to N=6140, Lxy=800 and H=100.
mkdir -p results

{
    bash tools/init.sh $K | ../poisson data.main > results/pos.dat 2> log;
} || {
    echo "ERROR: UAMMD Failed, here is the log" > /dev/stderr
    cat log > /dev/stderr
    exit 1
}
bash tools/computeDensityProfile.sh results/pos.dat $K > results/density.dat
bash tools/generateTheoreticalDensityProfile.sh $K > results/density.theo

H=$(grep -Eo "^H[[:space:]].*" data.main | awk '{print $2}')
meanError=$(paste results/den* | awk '$4&&$1<('$H'*0.5-4.0){m+=sqrt(($2-$4)**2)/$4;c++}END{print m/c}')
echo "Mean relative error (deviation from PNP theory far from wall): " $meanError
