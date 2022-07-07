
make Bonds

equilibriumDistance=1

cat<<EOF > init.pos
-5 0 0
5 0 0
EOF


cat<<EOF > particles.bonds
1
0 1 1 $equilibriumDistance
0
EOF

./Bonds data.main > pos.dat 2> log
finalDistance=$(tail -2 pos.dat | awk '{print $1, $2, $3}' | tr '\n' ' ' | awk '{print $4-$1, $5-$2, $6-$3}')

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
