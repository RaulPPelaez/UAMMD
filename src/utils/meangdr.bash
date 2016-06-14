

export OMP_NUM_THREADS=2
files=''
for i in {1..20}
do
    ./main > kk
    gdr -L 12.5 -rcut 6.25 $(ifile kk) -nbins 200  kk | tail -201 > gdr$i.dat
    
    files=$(echo "gdr$i.dat $files")
done

fileavg $files
