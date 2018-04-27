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
    ofstream out("data.main");
    out<<"#this is a comment!\n";
    out<<"option1 1      2.0   blabla 12342    \n";
    out<<"\n";
    out<<"option2\n";
    //It does not matter if the file ends just after an option
    out<<"option3";
    
  }
  InputFile in("data.main", sys);
  
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