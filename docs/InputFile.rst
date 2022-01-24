InputFile
===========

The InputFile class in utils/InputFile.h allows you to read parameters and options from a file.  

File format
------------

The input file must have the following format:
  
.. code:: bash
	  
  #Lines starting with '#' will be ignored
  [option] [argument1] [argument2] ...

  You can have an option with no arguments  
  Additionally you can use the special shell option, which will run the rest of the line as a bash command when encountered and wait for it to finish.  
  
Usage
---------

.. code:: cpp
	  
   InputFile inputFile("options.in");
   //Read an option of type float
   float cutOff;
   inputFile.getOption("cutOff")>>cutOff;
   
   //You can have several arguments for an option
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


getOption will return an `std::istringstream <http://www.cplusplus.com/reference/sstream/stringstream/>`_, so you can work with its output as such.

