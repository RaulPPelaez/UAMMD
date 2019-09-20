/*Raul P.Pelaez 2018

  An example on how to read parameters from a file.

  It uses the InputFile class [1].

  [1] https://github.com/RaulPPelaez/UAMMD/wiki/InputFile
 */

#include"uammd.cuh"
#include"utils/InputFile.h"
#include<fstream>
using namespace uammd;
using namespace std;
int main(){
  auto sys = make_shared<System>();

  {//Create the input file
    ofstream out("data.main.example");
    out<<"#this is a comment!\n";
    out<<"option1 1      2.0   blabla 12342    \n";
    out<<"\n";
    out<<"option2\n";
    //It does not matter if the file ends just after an option
    out<<"option3";

  }
  InputFile in("data.main.example", sys);

  //Order does not matter!
  if(!(in.getOption("option3"))){
    sys->log<System::ERROR>("option3 NOT FOUND!!!");
  }
  else{
    sys->log<System::MESSAGE>("Found option3!");
  }

  int a;
  float b;
  string c;
  int d;
  if(!(in.getOption("option1")>>a>>b>>c>>d))
    sys->log<System::ERROR>("OPTION NOT FOUND OR NOT ENOUGH PARAMETERS!!!");
  sys->log<System::MESSAGE>("Readed parameters for option %s: %d %f %s %d", "option1", a,b,c.c_str(), d);

  //Error if an option if not present!
  if(!(in.getOption("sdasds")))
    sys->log<System::ERROR>("sdsds NOT FOUND!!!");


    sys->finish();
  return 0;
};

//A copy pastable example, read parameters from a file, if it does not exist output a default one
real3 boxSize;
real dt;
std::string outputFile;
int numberParticles;
int numberSteps, printSteps;
void readParameters(std::shared_ptr<System> sys, std::string file){

  {
    if(!std::ifstream(file).good()){
      std::ofstream default_options(file);
      default_options<<"boxSize 32 32 32"<<std::endl;
      default_options<<"numberParticles 16384"<<std::endl;
      default_options<<"dt 0.001"<<std::endl;
      default_options<<"numberSteps 100000"<<std::endl;
      default_options<<"printSteps 100"<<std::endl;
      default_options<<"outputFile /dev/stdout"<<std::endl;
    }
  }

  InputFile in(file, sys);

  in.getOption("boxSize", InputFile::Required)>>boxSize.x>>boxSize.y>>boxSize.z;
  in.getOption("numberSteps", InputFile::Required)>>numberSteps;
  in.getOption("printSteps", InputFile::Required)>>printSteps;
  in.getOption("dt", InputFile::Required)>>dt;
  in.getOption("numberParticles", InputFile::Required)>>numberParticles;
  in.getOption("outputFile", InputFile::Required)>>outputFile;
}