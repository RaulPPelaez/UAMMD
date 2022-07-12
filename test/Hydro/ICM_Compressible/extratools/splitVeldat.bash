
rm -f vel.uammd.*
datamain=$1
time=$(awk '/^printTime/{print $2}' $datamain)
cat vel.dat | awk '/#/{p++;next}{t=(p-1)*'$time';print $0 > "vel.uammd."t}'
