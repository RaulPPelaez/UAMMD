
/*Raul P. Pelaez 2018. Input file parser.
  Allows to read parameters and options from a file.
  The input file must have the following format:

  #Lines starting with '#' will be ignored
  [option] [argument1] [argument2] ...
  
  You can have an option with no arguments

USAGE:

   //Creation:
   InputFile inputFile("options.in");

   //Read an option of type float
   float cutOff;
   inputFile.getOption("cutOff")>>cutOff;
   
   //You can have several arguments for an option
   real3 L;
   inputFile.getOption("boxSize")>>L.x>>L.y>>L.z;

   //Or none!
   bool isOptionPresent = bool(inputFile.getOption("someOption"));
   
   //You can check if an option is present in the file by casting to bool somehow
   //i.e
   if(!(inputFile.getOption("NotPresentOption"))){
     cerr<<"Option not present in the file!!!"<<endl;
   }

   //Finally you can check if the option not only exists but has the correct number/type of arguments
   if(!(inputFile.getOption("someOption")>>anInt>>aFloat>>aString)){
      cerr<<"Some parameter missing in the option!"<<endl;
   }
   


   getOption will return an std::istringstream, so you can work with its output as such.
   A second argument can be passed to getOption containing either InputFile::Required or InputFile::Optional (the latter being the default). If Required is passed and the option is not found, a CRITICAL log event will be issued and the program will terminate.

 */
#ifndef INPUT_FILE_H
#define INPUT_FILE_H


#include<string>
#include<System/System.h>
#include <sys/stat.h>
#include<fstream>
#include<vector>
#include<sstream>
#include"cxx_utils.h"
namespace uammd{
  class InputFile{

    shared_ptr<System> sys;
    size_t maxFileSizeToStore = 1e7; //10 mb
    std::string fileName;
    std::vector<std::pair<string,string>> options;
  public:
    enum OptionType{Required, Optional};
    
    InputFile(std::string name, shared_ptr<System> sys):fileName(name),
							sys(sys){
      struct stat stat_buf;
      int err = stat(name.c_str(), &stat_buf);
      if(err!=0) sys->log<System::CRITICAL>("[InputFile] ERROR: Could not open file %s!.", name.c_str());
      size_t fileSize = stat_buf.st_size;
      if(fileSize>=maxFileSizeToStore){
	sys->log<System::ERROR>("[InputFile] Attempting to store a file of size %s, are you sure you want to do this?", printUtils::prettySize(fileSize).c_str());
      }
      //Store options and arguments
      std::ifstream in(fileName);
      std::string line, tmp, word;     
      while(in>>word){
	//Ignore comments
	if(word.find_first_of("#")!=std::string::npos) getline(in, line);
	else{
	  //Given an option
	  in>>std::ws;
	  int next = in.peek();
	  //If the option has no arguments
	  if(next == '\n' || next == EOF){
	    options.emplace_back(std::make_pair(word, std::string()));
	    continue;
	  }
	  else{
	    //Otherwise store the rest of the line
	    getline(in,tmp);
	    options.emplace_back(std::make_pair(word, tmp));
	  }
	}
      
      }

    }

    std::stringstream getOption(std::string op, OptionType type = OptionType::Optional){
      sys->log<System::DEBUG>("[InputFile] Looking for option %s in file %s",  op.c_str(), fileName.c_str());
      for(auto s: options){
	if(std::get<0>(s).compare(op)==0){
	  sys->log<System::DEBUG>("[InputFile] Option found!");
	  std::stringstream ret(std::get<1>(s));	
	  return ret;
	}
      }
      sys->log<System::DEBUG>("[InputFile] Option not found!");
      if(type == OptionType::Required){
	sys->log<System::CRITICAL>("[InputFile] Option %s not found in %s!",op.c_str(), fileName.c_str());
      }
      std::stringstream bad_ss(std::string(""));
      bad_ss.setstate(std::ios::failbit);
      return  bad_ss;
    }

  };

}

#endif
