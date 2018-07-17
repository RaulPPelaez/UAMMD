#Computes pressure from rdf, expects the data.main that was used to generate the rdf data

L=$(grep boxSize data.main | awk '{print $2}')
N=$(grep numberParticles data.main | awk '{print $2}')
ep=$(grep epsilon data.main | awk '{print $2}')
sig=$(grep sigma data.main | awk '{print $2}')
T=$(grep temperature data.main | awk '{print $2}')
cutOff=$(grep cutOff data.main | awk '{print $2}')
rho=$(echo 1 | awk '{print '$N'/('$L'^3/('$sig'^3*1.0))}')

file=$1

dr=$(head -2 $file | awk '{print $1}' | tr '\n' ' ' | awk '{print $2-$1}')

#Uses a truncated potential, without shifting
cat $file |
    awk 'function ljdif(x){
    if(x>'$cutOff') return 0;
    else  return 24*'$ep'*'$sig'^6*(x^6-2*'$sig'^6)/x^13;
          }
          {
              print ljdif($1)*($1^3)*$2*'$dr'}' |
    awk '{sum+=$1}END{print sum;}' |
    awk '{print '$rho'*'$T'-(2/3.)*3.14151*'$rho'^2*$1}'






