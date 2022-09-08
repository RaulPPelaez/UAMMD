
make Bonds

function pairTest(){

    local equilibriumDistance=1

    cat<<EOF > init.pos
-5 0 0
5 0 0
EOF


    cat<<EOF > particles.bonds
1
0 1 1 $equilibriumDistance
0
EOF

    local posfile=pos.pair
    ./Bonds data.main > $posfile 2> log.pair
    finalDistance=$(tail -2 $posfile | awk '{print $1, $2, $3}' | tr '\n' ' ' | awk '{print $4-$1, $5-$2, $6-$3}')

    echo "Final distance between particles should be ($equilibriumDistance, 0, 0), found: $finalDistance"
    if echo $finalDistance |  awk '{exit(($1-'$equilibriumDistance')<1e-4?1:0)}'; then
	echo "Error in X!";
	exit 1
    else
	echo "X is Good!";
    fi
    if echo $finalDistance |  awk '{exit(($2)<1e-4?1:0)}';
    then
	echo "Error in Y!";
	exit 1
    else
	echo "Y is good!";
    fi
    if echo $finalDistance |  awk '{exit(($3)<1e-4?1:0)}';
    then
	echo "Error in Z!";
	exit 1
    else
	echo "Z is good!";
    fi
}



function chainTest(){
    local nParticles=10
    local dx=2
    local r0=1
    local posfile=pos.chain
    seq $nParticles | awk '{print $1*'$dx', 0,0}' > init.pos
    seq $nParticles | tail -n+2 |
	awk 'BEGIN{print '$nParticles-1'}{print NR-1, NR, 1, '$r0'}END{print 1; print "0 -3 0 -3 1 0";}' > particles.bonds

    ./Bonds data.main > $posfile 2> log.chain

}



pairTest
chainTest
