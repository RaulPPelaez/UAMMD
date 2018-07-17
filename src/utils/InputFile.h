
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
   //An additional parameter might be passed (Optional or Required, Optional by default) specifying if the option is necessary or not.
   //If required is specified and the option is not found, InputFile will issue a CRITICAL error
   real3 L;
   inputFile.getOption("boxSize", InputFile::Required)>>L.x>>L.y>>L.z;

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

    //Process a line of the file and store option/arguments if necessary
    //TODO: this could be prettier...
    void process_line(std::string &line){

      auto first_char = line.find_first_not_of(" \t\n");
      //Ignore empty lines
      if(first_char == std::string::npos) return;
      //Ignore comments
      if(line[first_char]=='#'){
	sys->log<System::DEBUG4>("[InputFile] Comment!");
	return;
      }

      else{
	std::string word;
	sys->log<System::DEBUG4>("[InputFile] Processing line!");
	//Given an option
	//in>>std::ws;
	auto first_non_space = line.find_first_not_of(" \t");
	line = line.substr(first_non_space, line.size());
	sys->log<System::DEBUG4>("[InputFile] remove left whitespaces: \"%s\"", line.c_str());
	auto space_after_option = line.find_first_of(" \t");
	if(space_after_option == std::string::npos){
	  word = line;
	  line = std::string();
	}
	else{	   
	  word = line.substr(0, space_after_option);	    
	  line = line.substr(space_after_option, line.size());
	  auto start_of_args = line.find_first_not_of(" \t\n");
	  if(start_of_args == std::string::npos){
	    line = std::string();
	  }
	  else{
	    line = line.substr(start_of_args, line.size());
	  }
	}
	sys->log<System::DEBUG3>("[InputFile] option \"%s\" registered with args \"%s\"",  word.c_str(), line.c_str());
	options.emplace_back(std::make_pair(word, line));	
      }
    }
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
      std::string line;
      while(!getline(in,line).eof()){
	process_line(line);
      }
      //process last line
      process_line(line);
    }

    
    //Returns a reference because g++-4.8 doesnt allow to std::move an stringstream... 
    std::istringstream& getOption(std::string op, OptionType type = OptionType::Optional){
      static std::istringstream ret;
      ret.str();
      ret.clear();
      sys->log<System::DEBUG1>("[InputFile] Looking for option %s in file %s",  op.c_str(), fileName.c_str());
      for(auto s: options){
	if(std::get<0>(s).compare(op)==0){
	  sys->log<System::DEBUG1>("[InputFile] Option found!");
	  //std::stringstream ret(std::get<1>(s));
	  ret.str(std::get<1>(s));
	  return ret;
	}
      }
      sys->log<System::DEBUG1>("[InputFile] Option not found!");
      if(type == OptionType::Required){
	sys->log<System::CRITICAL>("[InputFile] Option %s not found in %s!",op.c_str(), fileName.c_str());
      }
      //std::stringstream bad_ss(std::string(""));
      //bad_ss.setstate(std::ios::failbit);      
      //return  bad_ss;
      ret.setstate(std::ios::failbit);
      return ret;
      
    }

  };

}

#endif
