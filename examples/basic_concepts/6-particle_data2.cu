/* Raul P. Pelaez 2021
   Second part of the ParticleData tutorial.
   In this tutorial we will learn about particle ids and particle reordering.

 */

#include<uammd.cuh>
#include<random>
#include<thrust/random.h>
using namespace uammd;

int main(int argc, char* argv[]){
  //Lets create 1e5 particles (note that we are letting ParticleData take care of initializing System):
  const int numberParticles = 1e5;
  auto pd = std::make_shared<ParticleData>(numberParticles);
  //Lets start by placing the particles randomly
  {//We place particles randomly as in the previous tutorial.
    auto positions = pd->getPos(access::location::cpu, access::mode::write);
    std::mt19937 gen(pd->getSystem()->rng().next());
    std::uniform_real_distribution<real> dist(-0.5, 0.5);
    auto rng = [&](){return dist(gen);};
    real L = 64; //Lets randomly set positions inside a cubic box of size L
    std::generate(positions.begin(), positions.end(), [&](){ return make_real4(rng(), rng(), rng(), 0)*L;});
  }
  //And now lets learn about particle id
  {
    //ParticleData assigns each particle a name or numeric id at creation.
    //Id is treated as the rest of the properties, but you cannot modify it (so you can only requested it for reading)
    auto id = pd->getId(access::cpu, access::read);
    //After creation particle ids are the same as the index, so id[i]=i;
    //Lets print the first ten particles to check it:
    auto pos = pd->getPos(access::cpu, access::read);
    std::cout<<"Ten first particles after creation"<<std::endl;
    std::cout<<"Index\tName\tposition"<<std::endl;
    for(int i = 0; i<10; i++) std::cout<<i<<"\t"<<id[i]<<"\t"<<pos[i]<<std::endl;    
  }//As always, we delete the handles as soon as possible

  //If id[i]=i why is it that we need this "id" property at all?
  //ParticleData can reorder the particles in memory at any time, so that id[i] != i. It can decide to do this to increase the spatial locality of the data in memory, which can have a positive effect in performance. 
  //Usually this reorder is not an issue in GPU code, since many times we assign threads to particles and do not control the execution order anyway. But sometimes we need the initial order, for instance when writing particles to disk we would like to use the same order every time.
  //Lets see how to deal with this:
  //First lets force a reorder to happen:
  pd->sortParticles();
  {//If we print the names of the first particles again we will probably see that id[i] is no longer i
    auto id = pd->getId(access::cpu, access::read);
    auto pos = pd->getPos(access::cpu, access::read);
    std::cout<<"Ten first particles after reordering"<<std::endl;
    std::cout<<"Index\tName\tposition"<<std::endl;
    for(int i = 0; i<10; i++) std::cout<<i<<"\t"<<id[i]<<"\t"<<pos[i]<<std::endl;    
  }//You might notice that the printed particles are indeed close in space, cool, right? 
  //ParticleData can provide an array that stores the current index of a particle given its id:
  auto id2index = pd->getIdOrderedIndices(access::cpu);
  //Lets print the same thing now but using this indirection:
  {
    auto id = pd->getId(access::cpu, access::read);
    //Note that we could also refer to the id array as "index2id", the opposite indirection to id2index.
    auto pos = pd->getPos(access::cpu, access::read);
    std::cout<<"Particles with names from 0 to 9:"<<std::endl;
    std::cout<<"Index\tName\tposition"<<std::endl;
    for(int i = 0; i<10; i++){
      int index = id2index[i]; //The location (index) of particle with name (id) "i".
      std::cout<<i<<"\t"<<id[index]<<"\t"<<pos[index]<<std::endl;
    }
  }
  //The first two columns printed are once again equal and you can see that the positions are also the same as in the pre ordered state.
  //As a side note, it can be a PITA to write code that makes this indirection all the time, luckily we can make use of
  //thrust fancy iterators to write code that is transparent to this:
  //Lets print the same thing again using permutation_iterators:
  {
    auto id = pd->getId(access::cpu, access::read);
    auto pos = pd->getPos(access::cpu, access::read);
    //A permutation iterator takes an iterator and an index iterator and the indirection when accessed
    auto pos_by_id = thrust::make_permutation_iterator(pos.begin(), id2index); //pos_by_id[i] is now equivalent to pos[id2index[i]]
    //This indirection is completely useless,as the name suggest. By definition id_by_id[i] == i always. But so you get the idea.
    auto id_by_id = thrust::make_permutation_iterator(id.begin(), id2index);
    std::cout<<"Particles with names from 0 to 9:"<<std::endl;
    std::cout<<"Index\tName\tposition"<<std::endl;
    for(int i = 0; i<10; i++){
      std::cout<<i<<"\t"<<id_by_id[i]<<"\t"<<pos_by_id[i]<<std::endl;
    }
    //We can use these permutation iterators with any function that expects an iterator, either in the GPU or CPU (remember to choose the correct access to ParticleData in any case). In general, pos.begin() and pos_by_id as defined above present the same properties, so a templated function that takes one can also take the other. For instance, std::transform or thrust::transform.  
    //thrust provides a lot of convenient fancy iterators, look it up in the net, many times they will save you some headaches.
  }//The exact same thing gets printed again.
  
  //Destroy the UAMMD environment and exit
  pd->getSystem()->finish();
  return 0;
}
