/* Raul P. Pelaez 2021
   Your first UAMMD code.
   This file is the minimum file that can be considered UAMMD code.
   It will create an instance of System, the basic UAMMD structure that every other module will typically need.
   Among other startup operation, System sets up the CUDA environment and provides certain utilities as we will see in later examples.
   Note, however, that in general System is not required to be created explicitly, as we will see in the following examples.
 */

//uammd.cuh is the basic uammd include containing, among other things, the System struct.
#include<uammd.cuh>

//Everything UAMMD related comes under the uammd namespace, lets include it here to avoid writing uammd:: all the time
using namespace uammd;
//The main function will initialize an UAMMD environment, then destroy it and exit
int main(int argc, char* argv[]){
  //Many UAMMD modules (if not all) are stored in C++ shared pointers.
  //On creation, System will initialize the CUDA and UAMMD environments
  //System can read a few options from command line if provided with argc,argv. But lets ignore that for now.
  auto sys = std::make_shared<System>(argc, argv);
  //If succesfull some messages will be printed to the terminal with environment and UAMMD information.
  //Otherwise an error will be thrown and the execution will be halted.

  //Whenever UAMMD is not needed anymore it is a good idea to destroy any object related to it and then call sys->finish()
  // this will gracefully complete any pending cleanup operations.
  // Freeing all memory allocated and leaving the environment free to be used by any other code outside UAMMD.
  //Take into account that any UAMMD operation after this call will result in undefined behavior.
  sys->finish();
  return 0;
}
