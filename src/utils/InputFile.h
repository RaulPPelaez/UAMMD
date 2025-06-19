/*Raul P. Pelaez 2018-2021. Input file parser.
  Allows to read parameters and options from a file.
  The input file must have the following format:

  #Lines starting with '#' will be ignored
  [option] [argument1] [argument2] ...

  You can have an option with no arguments

  Additionally you can use the special shell option, which will run the rest of
the line as a bash command when encountered and wait for it to finish.

USAGE:

   //Creation:
   InputFile inputFile("options.in");

   //Read an option of type float
   float cutOff;
   inputFile.getOption("cutOff")>>cutOff;
   //An additional parameter might be passed (Optional or Required, Optional by
default) specifying if the option is necessary or not.
   //If required is specified and the option is not found, InputFile will issue
a CRITICAL error real3 L; inputFile.getOption("boxSize",
InputFile::Required)>>L.x>>L.y>>L.z;

   //Or none!
   bool isOptionPresent = bool(inputFile.getOption("someOption"));

   //You can check if an option is present in the file by casting to bool
somehow
   //i.e
   if(!(inputFile.getOption("NotPresentOption"))){
     cerr<<"Option not present in the file!!!"<<endl;
   }

   //Finally you can check if the option not only exists but has the correct
number/type of arguments
   if(!(inputFile.getOption("someOption")>>anInt>>aFloat>>aString)){
      cerr<<"Some parameter missing in the option!"<<endl;
   }

   getOption will return an std::istringstream, so you can work with its output
as such. A second argument can be passed to getOption containing either
InputFile::Required or InputFile::Optional (the latter being the default). If
Required is passed and the option is not found, an ERROR log event will be
issued and an std::runtime_error exception thrown.

 */
#ifndef INPUT_FILE_H
#define INPUT_FILE_H

#include "cxx_utils.h"
#include <System/System.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
namespace uammd {
class InputFile {

  size_t maxFileSizeToStore = 1e7; // 10 mb
  std::string fileName;
  std::vector<std::pair<string, string>> options;

  // Process a line of the file and store option/arguments if necessary
  void process_line(std::string &line) {
    auto first_char = line.find_first_not_of(" \t\n");
    // Ignore comments and empty lines
    if (first_char == std::string::npos or line[first_char] == '#') {
      return;
    } else {
      std::string word;
      auto first_non_space = line.find_first_not_of(" \t");
      line = line.substr(first_non_space, line.size());
      auto space_after_option = line.find_first_of(" \t");
      if (space_after_option == std::string::npos) {
        word = line;
        line = std::string();
      } else {
        word = line.substr(0, space_after_option);
        line = line.substr(space_after_option, line.size());
        auto start_of_args = line.find_first_not_of(" \t\n");
        if (start_of_args == std::string::npos) {
          line = std::string();
        } else {
          line = line.substr(start_of_args, line.size());
        }
      }
      System::log<System::DEBUG7>(
          "[InputFile] option \"%s\" registered with args \"%s\"", word.c_str(),
          line.c_str());
      if (word.compare("shell") == 0) {
        System::log<System::DEBUG3>("[InputFile] Executing shell command: %s",
                                    line.c_str());
        int rc = std::system(line.c_str());
        if (rc < 0) {
          System::log<System::ERROR>(
              "[InputFile] Shell command execution failed with code %d: %s", rc,
              line.c_str());
        }
        return;
      }
      options.emplace_back(std::make_pair(word, line));
    }
  }

public:
  enum OptionType { Required, Optional };

  InputFile(std::string name, shared_ptr<System> sys = nullptr)
      : fileName(name) {
    System::log<System::DEBUG4>("[InputFile] Reading from %s", name.c_str());
    struct stat stat_buf;
    int err = stat(name.c_str(), &stat_buf);
    if (err != 0) {
      System::log<System::ERROR>("[InputFile] ERROR: Could not open file %s!.",
                                 name.c_str());
      std::runtime_error("Parameter file not readable");
    }
    size_t fileSize = stat_buf.st_size;
    if (fileSize >= maxFileSizeToStore) {
      System::log<System::ERROR>("[InputFile] Attempting to store a file of "
                                 "size %s, are you sure you want to do this?",
                                 printUtils::prettySize(fileSize).c_str());
    }
    // Store options and arguments
    std::ifstream in(fileName);
    std::string line;
    while (!getline(in, line).eof()) {
      process_line(line);
    }
    // process last line
    process_line(line);
  }

  // Return a stringstream to the line of an option in the file (pointing to the
  // first argument) If the option does not exist and type==Required an
  // exception will be thrown Returns a reference because g++-4.8 doesnt allow
  // to std::move an stringstream...
  std::istringstream &getOption(std::string op,
                                OptionType type = OptionType::Optional) {
    static std::istringstream ret;
    ret.str();
    ret.clear();
    for (auto s : options) {
      if (std::get<0>(s).compare(op) == 0) {
        ret.str(std::get<1>(s));
        return ret;
      }
    }
    if (type == OptionType::Required) {
      System::log<System::ERROR>("[InputFile] Option %s not found in %s!",
                                 op.c_str(), fileName.c_str());
      throw std::runtime_error("Required option not found in file " + fileName);
    }
    ret.setstate(std::ios::failbit);
    if (op.compare("shell") == 0) {
      System::log<System::ERROR>(
          "[InputFile] Ignoring use of the reserved \"shell\" option");
    }
    return ret;
  }
};

} // namespace uammd

#endif
