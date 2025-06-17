/* Raul P. Pelaez 2021

   Hardcoding parameters in the source code becomes a problem quickly.
   It would be better if we read this parameters from a file at initialization.

   In this tutorial we will see a convenient object exposed by UAMMD called
   InputFile that does just that.

   You will see that you do not even need UAMMD for this, feel free to copy
   paste it into your project.
 */

#include "uammd.cuh"
#include "utils/InputFile.h"
using namespace uammd;

// Lets define some parameters like before, but this time lets not hardcode
// their values here We will read them from a parameter file
struct Parameters {
  int numberParticles;
  real3 boxSize;
  real viscosity;
  real dt;
  bool someOption;
  std::string someFileName;
};

// This function will read the file called datamain and fill and return a
// Parameters struct with it
Parameters readParameters(std::string datamain);

// Probably it is a good idea to write default parameters if the expected file
// does not exists
void writeDefaultParameters(std::string datamain);

int main() {

  std::string datamain = "data.main";

  // If the file does not exist print a default one
  if (not std::ifstream(datamain).good()) {
    writeDefaultParameters(datamain);
  }
  // Now take the parameters from the file
  auto par = readParameters(datamain);

  // Now par contains the parameters in the file data.main
  std::cout << par.boxSize << std::endl;

  // Try to remove some parameter from data.main and running again, if it is
  // marked as Required in readParameters and error will be thrown.
  return 0;
};

// this function will print a default parameter file to a file called datamain
void writeDefaultParameters(std::string datamain) {
  std::ofstream out(datamain);
  out << "#Lines starting with # will be ignored\n";
  out << "boxSize 32 32 32 #As will anything beyond the expected number of "
         "arguments\n";
  out << "numberParticles 128\n";
  out << "viscosity 1\n";
  out << "dt 1\n";
  out << "someFileName example.dat\n";
  out << "someOption\n";
  out << "#The special shell option will run the rest of the line and wait for "
         "completion when it is read by InputFile\n";
  out << "shell echo hello from data.main\n";
}

Parameters readParameters(std::string datamain) {
  InputFile in(datamain);
  Parameters par;
  // getOption returns a std::stringstream placed just after the the parameter
  // When the second argument is Required InputFile will terminate the program
  // with an error if the option in the first
  //  argument is not found,
  // You can pass Optional instead, which will silently return an empty
  // stringstream when the option is not present
  //  In this case the parameters will be left untouched by the >> operator.
  in.getOption("boxSize", InputFile::Required) >> par.boxSize.x >>
      par.boxSize.y >> par.boxSize.z;
  in.getOption("viscosity", InputFile::Required) >> par.viscosity;
  in.getOption("dt", InputFile::Required) >> par.dt;
  in.getOption("numberParticles", InputFile::Required) >> par.numberParticles;
  in.getOption("someFileName", InputFile::Required) >> par.someFileName;
  // You can have parameters without arguments and check if they exists or not
  // like this
  if (not in.getOption("someOption", InputFile::Optional)) {
    // Do something if the option is not present:
    par.someOption = false;
  }
  // You can also do just this:
  par.someOption = bool(in.getOption("someOption", InputFile::Optional));

  return par;
}
