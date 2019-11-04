/*Raul P.Pelaez 2017. Some system utilities

 */
#ifndef SYSUTILS_H
#define SYSUTILS_H
#include"global/defines.h"

#include<cstring>
#include<cstdlib>
#include<sstream>
namespace uammd{
  namespace input_parse{
    /*Returns the input argument number of a flag, -1 if it doesnt exist*/
    int checkFlag(int argc, char *argv[], const char *flag){
      fori(1, argc){
	if(strcmp(flag, argv[i])==0) return i;
      }
      return -1;
    }

    //Ask for one or many arguments of any type after a flag from argv
    template<class T>
    bool parseArgument(int argc, char *argv[],
		       const char *flag,  //What to look for
		       T* result, int numberArguments=1){ //output and number of them
      fori(1, argc){ //Look for the flag

	if(strcmp(flag, argv[i]) == 0){ //If found it
	  std::string line;  //convert char * to string for as many values as requested
	  forj(1,numberArguments)
	    line += argv[i+j];

	  std::istringstream ss(line);
	  //Store them in result
	  forj(0,numberArguments){
	    ss>>result[j];
	  }
	  return true;
	}
      }
      return false;
    }


  };

}
#endif
