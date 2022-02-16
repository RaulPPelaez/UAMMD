InputFile
===========

The InputFile class in utils/InputFile.h allows you to read parameters and options from a file.  


.. cpp:class:: InputFile

   .. cpp:function:: InputFile(std::string fileName);

      Initializes the class with a certain parameter file. See :ref:`file format` below.
      
   .. cpp:function:: std::istringstream getOption(std::string option, InputFile::OptionType type );

      :param option: The option to look for in the file.
      :param type: Whether the option is mandatory or optional. An exception will be thrown if a mandatory option is not found in the file.
      :return: A stream with the contents of the line containing the option (not including the option). Empty if the option was not found.
      
   .. cpp:enum:: OptionType

      .. cpp:enumerator:: \
			  Optional
			  Required

			  

File format
------------

The input file must have the following format:
  
.. code:: bash
	  
  #Lines starting with '#' will be ignored
  #[option] [argument1] [argument2] ...

  exampleOption argument1 argument2
  #You can have an option with no arguments
  justAFlag
  #Additionally you can use the special shell option, which will run the rest of the line as a bash command when encountered and wait for it to finish.  
  shell bash run.sh
  shell echo "Hello"

  
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
   int anInt;
   float aFloat;
   std::string aString;
   if(!(inputFile.getOption("someOption")>>anInt>>aFloat>>aString)){
      cerr<<"Some parameter missing in the option!"<<endl;
   }


.. note:: getOption will return an `std::istringstream <http://www.cplusplus.com/reference/sstream/stringstream/>`_, so you can work with its output as such.

