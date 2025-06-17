

#include "Integrator/Hydro/BDHI_quasi2D.cuh"
#include "uammd.cuh"

#include "utils/InputFile.h"
#include <cstring>
#include <fstream>

using namespace uammd;

using std::cerr;
using std::endl;
using std::make_shared;

int N;
real3 boxSize = make_real3(0);

int2 cells = {-1, -1};
std::string initFile, outFile("/dev/stdout");

real viscosity, temperature, dt, tolerance = 1e-3, hydrodynamicRadius;

int numberSteps, printSteps, relaxSteps;

std::string scheme;

int in_numberParticles;
bool loadParticles = true;

void readParameters(std::string fileName);
int main(int argc, char *argv[]) {
  auto sys = make_shared<System>(argc, argv);
  ullint seed = 0xf31337Bada55D00dULL ^ time(NULL);
  sys->rng().setSeed(seed);
  readParameters(sys->getargv()[1]);
  Box box(boxSize);
  std::shared_ptr<ParticleData> pd;
  if (loadParticles) {
    std::ifstream in(initFile);
    if (!in)
      sys->log<System::CRITICAL>("Could not read from %s!", initFile.c_str());
    in >> N;
    pd = make_shared<ParticleData>(N, sys);
    {
      auto pos = pd->getPos(access::location::cpu, access::mode::write);
      fori(0, N) {
        in >> pos[i].x >> pos[i].y >> pos[i].z;
        pos[i].w = 0;
      }
    }
  } else {
    N = in_numberParticles;
    if (!N)
      sys->log<System::CRITICAL>(
          "I need numberParticles if loadParticles == 0 ");
    pd = make_shared<ParticleData>(N, sys);
    auto pos = pd->getPos(access::location::cpu, access::mode::write);
    fori(0, N) {
      pos[i] = make_real4(
          make_real3(sys->rng().uniform3(-boxSize.x * 0.5, boxSize.x * 0.5)));
      pos[i].z = 0;
    }
  }
  auto pg = make_shared<ParticleGroup>(pd, "All");
  std::ofstream out(outFile);
  std::shared_ptr<Integrator> bdhi;
  if (scheme.compare("quasi2D") == 0) {
    using Scheme = BDHI::Quasi2D;
    Scheme::Parameters par;
    par.temperature = temperature;
    par.viscosity = viscosity;
    par.dt = dt;
    par.hydrodynamicRadius = hydrodynamicRadius;
    par.cells = cells;
    par.box = box;
    bdhi = make_shared<Scheme>(pg, par);
  } else if (scheme.compare("true2D") == 0) {
    using Scheme = BDHI::True2D;
    Scheme::Parameters par;
    par.temperature = temperature;
    par.viscosity = viscosity;
    par.dt = dt;
    par.hydrodynamicRadius = hydrodynamicRadius;
    par.cells = cells;
    par.box = box;
    bdhi = make_shared<Scheme>(pg, par);
  } else {
    sys->log<System::CRITICAL>("Unrecognized scheme!, use quasi2D or ture2D");
  }
  sys->log<System::MESSAGE>("RUNNING!!!");
  Timer tim;
  tim.tic();
  int nsteps = numberSteps;
  forj(0, relaxSteps) { bdhi->forwardTime(); }
  forj(0, nsteps) {
    bdhi->forwardTime();
    if (printSteps > 0 and j % printSteps == 0) {
      sys->log<System::DEBUG1>("[System] Writing to disk...");
      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      const int *sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
      out << "#" << "\n";
      fori(0, N) {
        real4 pc = pos[sortedIndex[i]];
        real3 p = make_real3(pc);
        int type = pc.w;
        real radius = hydrodynamicRadius;
        out << p << " " << radius << " " << type << "\n";
      }
      out << std::flush;
    }
  }
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps / totalTime);
  sys->finish();
  return 0;
}

void readParameters(std::string fileName) {
  InputFile in(fileName);

  in.getOption("boxSize", InputFile::Required) >> boxSize.x >> boxSize.y;
  in.getOption("cells", InputFile::Optional) >> cells.x >> cells.y;
  in.getOption("numberSteps", InputFile::Required) >> numberSteps;
  in.getOption("printSteps", InputFile::Required) >> printSteps;
  in.getOption("dt", InputFile::Required) >> dt;
  in.getOption("relaxSteps", InputFile::Required) >> relaxSteps;
  in.getOption("viscosity", InputFile::Required) >> viscosity;
  in.getOption("temperature", InputFile::Required) >> temperature;
  in.getOption("hydrodynamicRadius", InputFile::Required) >> hydrodynamicRadius;

  in.getOption("output", InputFile::Optional) >> outFile;
  in.getOption("tolerance", InputFile::Optional) >> tolerance;
  in.getOption("scheme", InputFile::Required) >> scheme;

  in.getOption("initcoords", InputFile::Optional) >> initFile;
  in.getOption("numberParticles", InputFile::Optional) >> in_numberParticles;
  in.getOption("loadParticles", InputFile::Optional) >> loadParticles;
}
