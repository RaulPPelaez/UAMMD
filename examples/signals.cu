/*Raul P. Pelaez 2021. Example showing how to connect to ParticleData signals.

  Say the class called "User" wants to know when some particle property has been modified. For example because it needs to keep something up to date when the positions are modified. 
  The class "User" might be an Integrator, Interactor, a Potential, a Transverser... or anything outside the UAMMD framework.  

 */
#include"uammd.cuh"

using namespace uammd;
using std::endl;
using std::make_shared;

//For some reason this class needs to know when someone accessed
//the positions and/or the velocities of the particles with the intention of writting to them.
class User{
  //A connection must be used to keep track of the signal
  connection positionConnection;
  connection velocityConnection;
  //Internal variables needed to perform some arbitrary computation
  bool pos_needs_processing = true;
  bool vel_needs_processing = true;
  std::shared_ptr<ParticleData> pd;
public:
  User(std::shared_ptr<ParticleData> pd):pd(pd){
    //Here we bind some member functions to the related signals.
    //When the signal emits the bound function will be called 
      positionConnection = pd->getPosWriteRequestedSignal()->connect([this](){this->handle_pos_access();});
      velocityConnection = pd->getVelWriteRequestedSignal()->connect([this](){this->handle_vel_access();});
  }
  ~User(){
    //Remember to disconnect when the signal is not needed anymore.
    positionConnection.disconnect();
    velocityConnection.disconnect();
  }
  //Say this function needs to be sure that it has processed the current positions and velocities.
  //Instead of assuming both have changed everytime the function is called we can rely on the signals
  // to do work only when any of them changed
  //Say this variable is the result of some operation applied to positions in .x and velocities in .y
  real2 currentResults = {-1,-1};
  real2 some_work_using_current_pos_and_vel(){
    //No work needs to be done if neither pos nor vel were modified since last call
    if(not pos_needs_processing and not vel_needs_processing){
      System::log<System::MESSAGE>("[User] No work needs to be done");
      return currentResults;
    }
    //Only do work with positions if we know it has been potentially modified since last call
    if(pos_needs_processing){
      System::log<System::MESSAGE>("[User] Doing position-dependent computation");
      //Be careful with writing to pos here, as it will result in the signal emitting again and thus beating the purpose
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      //The center of mass in X, just because
      currentResults.x = std::accumulate(pos.begin(), pos.end(), real4()).x/pd->getNumParticles();
      //No work needs to be done until the next time positions change 
      pos_needs_processing = false;
    }
    //Only do work with velocities if we know it has been potentially modified since last call
    if(vel_needs_processing){
      System::log<System::MESSAGE>("[User] Doing velocity-dependent computation");
      auto vel = pd->getVel(access::location::cpu, access::mode::read);
      //The total velocity in X, just because
      currentResults.y = std::accumulate(vel.begin(), vel.end(), real3()).x;
      //No work needs to be done until the next time velocities change 
      vel_needs_processing = false;
    }
    //Notice that if this function is called two times, without modifying pos or vel in between, the second time
    // will perform no work and simply return the stored value
    return currentResults;
  }

private:
  //This function will be called when pd->getPos() is called with access::mode::write or readwrite.
  void handle_pos_access(){
    //Notice that this function is called when the property has been requested, so when this function is called
    //The positions have not been modified yet.
    //Still you can use this slot function to set up some flag:
    pos_needs_processing = true;
    System::log<System::MESSAGE>("[USER] Positions was requested for writting");       
  }
  //This function will be called when pd->getVel() is called with access::mode::write or readwrite.
  void handle_vel_access(){
    //Same as with handle_pos_access
    vel_needs_processing = true;
    System::log<System::MESSAGE>("[USER] Velocity was requested for writting");
  }

};


void modify_positions(std::shared_ptr<ParticleData> pd){
  auto pos = pd->getPos(access::location::cpu, access::mode::write);
  pos[0] = {1,0,0,0};
  pos[1] = {3,0,0,0};
}

void modify_velocities(std::shared_ptr<ParticleData> pd){
  auto vel = pd->getVel(access::location::cpu, access::mode::write);  
  vel[0] = {1,0,0};
  vel[1] = {-2,0,0};
}

//Try to run this and see the output messages, you will see something like this:
int main(int argc, char *argv[]){
  //Initialize uammd
  const int N = 2;
  auto sys = std::make_shared<System>(argc, argv);
  auto pd = std::make_shared<ParticleData>(N, sys);
  {//Initialize with some irrelevant values
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    pos[0] = {1,0,0,0};
    pos[1] = {2,0,0,0};
    auto vel = pd->getVel(access::location::cpu, access::mode::write);
    vel[0] = {1,0,0};
    vel[1] = {-2,0,0};
  }
  //User construction will connect to pd such that it will be informed when positions or velocities are requested for writting.  
  User user(pd);
  real2 res;
  //Work will be done the first time
  res = user.some_work_using_current_pos_and_vel();
  //Now no work will be performed, since neither pos nor vel have been modified
  res = user.some_work_using_current_pos_and_vel();
  //Write something to pos
  modify_positions(pd);
  //When pd->getPos(...write) was called the "user" instance was informed via signals and User::handle_pos_access was called
  //Now some work must be done to take into account the access to position
  res = user.some_work_using_current_pos_and_vel();
  //No work needs to be done this time
  res = user.some_work_using_current_pos_and_vel();
  //write something to pos
  modify_positions(pd);
  //write something to vel
  modify_velocities(pd);
  //Work will be done to take into account new pos and vel
  res = user.some_work_using_current_pos_and_vel();
  
  //this line is here to silence the "unused variable" warning when compiling
  [&res]{}();
  sys->finish();
  return 0;
}
