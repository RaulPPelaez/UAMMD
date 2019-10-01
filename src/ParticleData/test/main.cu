

#include"uammd.cuh"

#include<random>
#include<algorithm>

using namespace std;
using namespace uammd;

class User{
public:
  User(std::shared_ptr<ParticleData> pd){
    pd->getReorderSignal()->connect(std::bind(&User::handle_reorder, this));
  }
  void handle_reorder(){
    std::cout<<"A sorting occured!"<<std::endl;
  }
private:

};

int main(){
  cout<<"Starting"<<endl;
  int N = 100;

  auto sys = std::make_shared<System>();
  auto pd = std::make_shared<ParticleData>(N, sys);

  cerr<<"Filling  CPU..."<<endl;
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(0,1);
  {
    auto poscpuf = pd->getPos(access::cpu, access::write);
    //std::generate((float*)poscpuf.raw(), (float*)poscpuf.raw()+4*N,  [&](){return dist(gen)*100;});
    for(int i=0; i<N; i++) poscpuf.raw()[i] = make_real4(N-i, N-i, N-i, rand()%2);

    cerr<<"DONE!"<<endl;
  }
  cerr<<"Printing initial pos..."<<endl;
  {
    auto poscpu= pd->getPos(access::cpu, access::read);
    auto  idcpu= pd->getId(access::cpu, access::read);
    cerr<<"Position x:"<<endl;
    for(int i=0; i<N; i++) cout<<poscpu.raw()[i].x<<" ";
    cout<<endl;
    cerr<<"ID:"<<endl;
    for(int i=0; i<N; i++) cout<<idcpu.raw()[i]<<" ";
    cout<<endl;
  }

  auto selector = particle_selector::IDRange(4, 8);

  auto pg = make_shared<ParticleGroup>(selector, pd, sys, "Group0");

  cerr<<"Printing particle Group:"<<endl;

  {
    auto poscpu= pd->getPos(access::cpu, access::read);
    auto  idcpu= pd->getId(access::cpu, access::read);

    auto indices = pg->getIndexIterator(access::cpu);
    int n = pg->getNumberParticles();
    cerr<<"Particles in group: "<<endl;
    for(int i=0; i<n; i++) cout<<indices[i]<<" ";
    cout<<endl;
    cerr<<"Position x:"<<endl;
    for(int i=0; i<n; i++) cout<<poscpu.raw()[indices[i]].x<<" ";
    cout<<endl;
    cerr<<"ID:"<<endl;
    for(int i=0; i<n; i++) cout<<idcpu.raw()[indices[i]]<<" ";
    cout<<endl;

  }


  User user(pd);
  cout<<"Emitting sort signal.."<<endl;
  pd->sortParticles();

   cerr<<"Printing sorted pos..."<<endl;
   {
     auto   poscpu= pd->getPos(access::cpu, access::read);
     auto   idcpu= pd->getId(access::cpu, access::read);
    cerr<<"Position x:"<<endl;
    for(int i=0; i<N; i++) cout<<poscpu.raw()[i].x<<" ";
    cout<<endl;
    cerr<<"ID:"<<endl;
    for(int i=0; i<N; i++) cout<<idcpu.raw()[i]<<" ";
    cout<<endl;
   }
   cudaDeviceSynchronize();

  cerr<<"Printing particle Group:"<<endl;

  {
    auto poscpu= pd->getPos(access::cpu, access::read);
    auto  idcpu= pd->getId(access::cpu, access::read);

    auto indices = pg->getIndexIterator(access::cpu);
    int n = pg->getNumberParticles();

    cerr<<"Particles in group: "<<endl;
    for(int i=0; i<n; i++) cout<<indices[i]<<" ";
    cout<<endl;
    cerr<<"Position x:"<<endl;
    for(int i=0; i<n; i++) cout<<poscpu.raw()[indices[i]].x<<" ";
    cout<<endl;
    cerr<<"ID:"<<endl;
    for(int i=0; i<n; i++) cout<<idcpu.raw()[indices[i]]<<" ";
    cout<<endl;
  }




   /*
   cerr<<"Adding particles"<<endl;

   pd->changeNumParticles(N+10);
   {
     auto poscpuf = pd->getPos(access::cpu, access::write);
     std::generate((float*)poscpuf.raw()+4*N, (float*)poscpuf.raw()+4*N+4*10,  [&](){return dist(gen)*100;});

     auto idcpu = pd->getId(access::cpu, access::write);

     for(int i=0; i<10; i++){
       idcpu.raw()[N+i] = N+i;
     }

   }
   N = pd->getNumParticles();
   {
     auto   poscpu= pd->getPos(access::cpu, access::read);
     auto   idcpu= pd->getId(access::cpu, access::read);

     for(int i=0; i<N; i++) cout<<poscpu.raw()[i].x<<" ";
     cout<<endl;
     for(int i=0; i<N; i++) cout<<idcpu.raw()[i]<<" ";
     cout<<endl;
   }
   cout<<"Emitting.."<<endl;
   pd->sortParticles();
   cerr<<"Printing sorted pos..."<<endl;
   {
     auto   poscpu= pd->getPos(access::cpu, access::read);
     auto   idcpu= pd->getId(access::cpu, access::read);
     for(int i=0; i<N; i++) cout<<poscpu.raw()[i].x<<" ";
     cout<<endl;
     for(int i=0; i<N; i++) cout<<idcpu.raw()[i]<<" ";
     cout<<endl;
   }
   cudaDeviceSynchronize();

   cerr<<"Removing particles"<<endl;
   pd->changeNumParticles(N-10);
   {
     auto poscpuf = pd->getPos(access::cpu, access::write);
     std::generate((float*)poscpuf.raw()+4*N, (float*)poscpuf.raw()+4*N+4*10,  [&](){return dist(gen)*100;});

     auto idcpu = pd->getId(access::cpu, access::write);

     for(int i=0; i<10; i++){
       idcpu.raw()[N+i] = N+i;
     }

   }
   N = pd->getNumParticles();
   {
     auto   poscpu= pd->getPos(access::cpu, access::read);
     auto   idcpu= pd->getId(access::cpu, access::read);

     for(int i=0; i<N; i++) cout<<poscpu.raw()[i].x<<" ";
     cout<<endl;
     for(int i=0; i<N; i++) cout<<idcpu.raw()[i]<<" ";
     cout<<endl;
   }
   cout<<"Emitting.."<<endl;
   pd->sortParticles();
   cerr<<"Printing sorted pos..."<<endl;
   {
     auto   poscpu= pd->getPos(access::cpu, access::read);
     auto   idcpu= pd->getId(access::cpu, access::read);
     for(int i=0; i<N; i++) cout<<poscpu.raw()[i].x<<" ";
     cout<<endl;
     for(int i=0; i<N; i++) cout<<idcpu.raw()[i]<<" ";
     cout<<endl;
   }



   */



   cudaDeviceSynchronize();
   pd.reset();

  // float4 *d_m, *d_out;
  // int n= 100;
  // int ntest=500000;

  // cudaMalloc(&d_m, n*sizeof(float4));
  // cudaMalloc(&d_out, n*sizeof(float4));
  // cudaMemset(d_m, 0, n*sizeof(float4));

  // cub::TexObjInputIterator<float4> tex;
  // tex.BindTexture(d_m, n*sizeof(float4));
  // tex.UnbindTexture();
  // cudaDeviceSynchronize();
  // cerr<<"Binding test"<<endl;
  // auto t_start = std::chrono::high_resolution_clock::now();
  // for(int i=0; i<ntest;i++){
  //   tex.BindTexture(d_m, n*sizeof(float4));
  //   transverse<<<n/128, 128>>>(tex, n, d_out);
  //   tex.UnbindTexture();
  // }
  // auto t_end = std::chrono::high_resolution_clock::now();

  // cerr<< std::chrono::duration<double, std::milli>(t_end-t_start).count()/ntest<<"ms"<<endl;

  // cerr<<"No binding test"<<endl;
  // t_start = std::chrono::high_resolution_clock::now();
  // tex.BindTexture(d_m, n*sizeof(float4));
  // for(int i=0; i<ntest;i++)
  //   transverse<<<n/128, 128>>>(tex, n, d_out);

  //   t_end = std::chrono::high_resolution_clock::now();
  // cerr<< std::chrono::duration<double, std::milli>(t_end-t_start).count()/ntest<<"ms"<<endl;

  // print<<<5,1>>>((float*)d_out,5);

  cudaDeviceSynchronize();
  return 0;
}
