Programming Tools
===================

Pointers and memory addresses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Smart Pointers
----------------

Instead of using raw C-style pointers, when creating references UAMMD usually uses c++ smart pointers. Specifically most objects are created and shared using `shared_ptr <https://en.wikipedia.org/wiki/Smart_pointer>`_.  
shared_ptr handles the creation and destruction of the object (replaces malloc/free new/delete). The interesting thing about shared_ptr is that the object lives until the last shared_ptr to it lives.

Example
********

.. code:: cpp
	  
  #include<memory>
  int main(){
      std::shared_ptr<float> f1, f2;
      //f1 and f2 are == to nullptr
      f1 = std::make_shared<float>(5.0f);
      // *f1 contains 5.0f and is != nullptr. f2 == nullptr
      f2 = f1;
      //f2 points to the same location as f1
      f1.reset();
      //f1 is destroyed, now f1==nullptr, but f2 has not changed
      //*f2 still contains 5.0f
      f2.reset();
      //Now f2, the last pointer to the float has been destroyed and now the memory storing the float is freed
      return 0;
   }


****

Structures
~~~~~~~~~~~

Functor
-----------

A functor is a struct trick to emulate a "function object". A functor is just a regular struct, but the parenthesis operator, "()", is overloaded to perform an arbitrary computation.  
The benefit of a functor is that it can be passed around as a regular struct, or have it as a template argument.  
An example of a functor:

.. code:: cpp
	  
  /*A minimal functor*/
  /*Just a regular struct*/
  struct my_functor{
    inline __device__ __host__ void operator()(/*no inputs*/){
      printf("Hello functor!\n");
    }
  };
  
  /*A more complex functor*/
  struct my_functor2{
    int a_member; /*A functor can have members and even constructors and other methods*/
    my_functor2(){ a_member = 10} 
    inline __device__ __host__ real operator()(real a, real b, .../*Any number of input arguments*/){
        return a*b; /*Perform some computation with the inputs*/
    }
  };


In UAMMD, functors are usually used as template parameters to outsource some key computation. Throughout UAMMD, the term functor is abused to denote simply a struct with some key public methods (like a kind of interface).  
For example ExternalForces needs a struct with a force and/or energy methods and does nothing with the parenthesis operator, still I call this a functor.  

*****      

.. ## Object Oriented Programming  
.. 
.. ## Virtual classes and inheritance  
.. 
.. ## Template Oriented Programming  

SFINAE
~~~~~~~~~

Substitution Failure Is Not An Error.  
This template technique abuses the default behavior of C++ template metaprogramming when a template overload substitution fails. In this case, instead of producing a compiler error, the compiler will simply discard the overload and go for the next candidate.  
There are numerous ways to abuse this behavior, but UAMMD mainly uses it to provide a default behavior when some method is missing in a functor.  
For example, if ExternalForces is used with a functor that does not implement an energy function, ExternalForces will just assume that the energy is 0 (or not needed).  
You can see some SFINAE examples in utils/cxx_utils.h or utils/TransverserUtils.cuh.  

.. ### Variadic templates  
