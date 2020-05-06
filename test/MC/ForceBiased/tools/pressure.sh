#Computes pressure from rdf, expects the data.main that was used to generate the rdf data


N=$(grep numberParticles data.main | awk '{print $2}')
ep=$(grep epsilon data.main | awk '{print $2}')
sig=$(grep sigma data.main | awk '{print $2}')
l=$(grep boxSize data.main | awk '{print $2/'$sig'}')
t=$(grep temperature data.main | awk '{print $2/'$ep'}')
cutOff=$(grep cutOff data.main | awk '{print $2}')
rho=$(echo 1 | awk '{print '$N'/('$l'^3)}')

file=$1

dr=$(head -2 $file | awk '{printf "%.13g\n", $1/'$sig'}' | tr '\n' ' ' | awk '{printf "%.13g", $2-$1}')

#Uses a truncated potential, without shifting
cat $file | awk '{printf "%.13g %.13g %.13g\n", $1/'$sig', $2, $3}' |
    awk 'function ljdif(x){
    if(x>'$cutOff' || x<=1e-10 ) return 0;
    else{
      return 24*(x^6-2)/(x)^13;
      }
    }
    {print ljdif($1)*($1^3)*$2*'$dr'}' |
    awk '{sum+=$1}END{printf "%.13g", sum;}' |
    awk '{printf "%.13g", '$rho'*'$t'-(2/3.)*3.14151*'$rho'^2*$1}'






