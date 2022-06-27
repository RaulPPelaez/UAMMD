#Raul P. Pelaez 2022. Test script for the compressible ICM.
#Runs every other test
set -e

ndev=1 #Number of available GPUs
tar xvf tools.tar.xz #Unpack the theory data
bash structureFactorTest.bash $ndev
bash particleDiffusionTest.bash $ndev
bash radialDistributionFunction $ndev
