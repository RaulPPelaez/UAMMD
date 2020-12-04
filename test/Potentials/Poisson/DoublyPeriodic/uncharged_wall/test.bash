
if ! test -f data.main; then echo "ERROR: This script expects a data.main to be present" >/dev/stderr; exit 1; fi
if ! test -f ../poisson; then echo "ERROR: This script poisson to be compiled and available at ../ " >/dev/stderr; exit 1; fi

mkdir -p results

{
    bash tools/init.sh | ../poisson data.main > results/pos.dat 2> log;
} || {
    echo "ERROR: UAMMD Failed, here is the log" > /dev/stderr
    cat log > /dev/stderr
    exit 1
}
bash tools/computeDensityProfile.sh results/pos.dat > results/density.dat
