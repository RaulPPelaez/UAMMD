/*WORK IN PROGRESS
  Raul P.Pelaez 2016- Simulation Script
   
   Implementation of the SimulationScript class. 

   The constructor of this class reads a simulation configuration from a script, constructs it and runs it.
 */

#include "SimulationScript.h"
#include<random>
#include<iomanip>
#include<memory>
using namespace std;

SimulationScript::SimulationScript(int argc, char* argv[], const char *fileName): Driver(){
  
  Timer tim; tim.tic();

  setUpMaps();
  
  cerr<<tim.toc()<<endl;
}


void SimulationScript::setUpMaps(){

  // parameterMap["Temperature"] = 0;

  // set(option, values);
  
  // type = typeMap[option];
  // vector<string> args = stringSplit(values);  
  // switch(type){
  // case PARAMETERS:
  //   if(args.size()!=1){
  //     cerr<<"Reading error!!, wrong option in "<<option<<endl;
  //     exit(1);
  //   }
  //   *parameterMap[option] = stod(args[0]);
  //   break;
  // case INTERACTORS:
    
  //   break;
  
    
  // }
    

  


  

}
