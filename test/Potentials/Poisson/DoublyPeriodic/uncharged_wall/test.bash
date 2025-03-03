
if ! test -f data.main; then echo "ERROR: This script expects a data.main to be present" >/dev/stderr; exit 1; fi
if ! test -f data.main.relax; then echo "ERROR: This script expects a data.main.relax to be present" >/dev/stderr; exit 1; fi
if ! test -f ../dppoisson; then echo "ERROR: This script poisson to be compiled and available at ../ " >/dev/stderr; exit 1; fi

mkdir -p results

{
    numberParticles=$(grep -Eo "^numberParticles[[:space:]].*" data.main | awk '{print $2}')
    bash tools/init.sh | ../dppoisson data.main.relax 2> log.relax | tail -$numberParticles | ../dppoisson data.main > results/pos.dat 2> log;
} || {
    echo "ERROR: UAMMD Failed, here is the log" > /dev/stderr
    cat log.relax log > /dev/stderr
    exit 1
}
bash tools/computeDensityProfile.sh results/pos.dat > results/density.dat
