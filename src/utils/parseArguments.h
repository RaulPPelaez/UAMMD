/*Raul P. Pelaez 2018
  Some utilities to process argv and argc
*/
#ifndef PARSEARGUMENTS_H
#define PARSEARGUMENTS_H

#include<sstream>
namespace uammd{
  //Ask for one or many arguments of any type after a flag from argv. returns true if numberArguments where found after flag
  namespace input_parse{
    template<class T>
    bool parseArgument(int argc, char *argv[],
		       const char *flag,  //What to look for
		       T* result, int numberArguments=1){ //output and number of them
      for(int i=1; i<argc; i++){ //Look for the flag

	if(strcmp(flag, argv[i]) == 0){ //If found it
	  for(int j=0; j<numberArguments; j++)  result[j] = T();
	  std::string line;  //convert char * to string for as many values as requested
	  if(argc<i+numberArguments+1) return false;
	  for(int j=0; j<numberArguments; j++){
	    line += argv[i+j+1];
	  }
	  std::istringstream ss(line);
	  //Store them in result
	  for(int j=0; j<numberArguments; j++){

	    ss>>result[j];
	  }
	  return true;
	}
      }
      return false;
    }
    //Returns true if the flag is present in argv, flase otherwise
    bool checkFlag(int argc, char *argv[],
		   const char *flag){  //What to look for
      for(int i=1; i<argc; i++){ //Look for the flag
	if(strcmp(flag, argv[i]) == 0){ //If found it
	  return true;
	}
      }
      return false;
    }
  }
}
#endif
