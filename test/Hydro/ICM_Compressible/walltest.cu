/* Raul P. Pelaez 2022. Compressible ICM test code.

   Imposes wall boundary conditions at the bottom and top walls.
   We impose an oscillatory velocity (in x) on the bottom wall.
   Prints the fluid velocity in the x direction vs z at y=0 for several times.
 */
#include "Integrator/Hydro/ICM_Compressible.cuh"
#include "Integrator/Integrator.cuh"
#include "misc/ParameterUpdatable.h"
#include "uammd.cuh"
#include "utils/InputFile.h"
#include "utils/container.h"
#include <cstdint>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace uammd;

// Boundary conditions for ICM (can only be set in Z).
// Places a fixed wall at the top of the domain and a moving one at the bottom.
class ICMWalls : public ParameterUpdatable {
  real currentTime = 0;
  real bottomWallvx = 0;
  real freq;
  real amplitude;
  using FluidPointers = Hydro::icm_compressible::FluidPointers;

public:
  ICMWalls() {}
  ICMWalls(real amplitude, real freq = 0) : amplitude(amplitude), freq(freq) {}

  // Returns wether there are walls in the Z direction. If false the Z domain
  // ghost cells are periodic.
  __host__ __device__ bool isEnabled() { return true; }

  // Applies the boundary conditions at the bottom z wall for the fluid
  __device__ void applyBoundaryConditionZBottom(FluidPointers fluid,
                                                int3 ghostCell, int3 n) const {
    const int ighost =
        ghostCell.x + (ghostCell.y + ghostCell.z * (n.y + 2)) * (n.x + 2);
    // The index of the cell above the ghost cell
    const int ighostZp1 =
        ghostCell.x + (ghostCell.y + (ghostCell.z + 1) * (n.y + 2)) * (n.x + 2);
    real rho = fluid.density[ighostZp1];
    fluid.density[ighost] = rho;
    fluid.velocityX[ighost] = 2 * bottomWallvx - fluid.velocityX[ighostZp1];
    fluid.velocityY[ighost] = -fluid.velocityY[ighostZp1];
    fluid.velocityZ[ighost] = -fluid.velocityZ[ighostZp1];
    fluid.momentumX[ighost] =
        2 * bottomWallvx * rho - fluid.momentumX[ighostZp1];
    fluid.momentumY[ighost] = -fluid.momentumY[ighostZp1];
    fluid.momentumZ[ighost] = -fluid.momentumZ[ighostZp1];
  }

  // Applies the boundary conditions at the top z wall for the fluid
  __device__ void applyBoundaryConditionZTop(FluidPointers fluid,
                                             int3 ghostCell, int3 n) const {
    const int ighost =
        ghostCell.x + (ghostCell.y + ghostCell.z * (n.y + 2)) * (n.x + 2);
    // The index of the cell below the ghost cell
    const int ighostZm1 =
        ghostCell.x + (ghostCell.y + (ghostCell.z - 1) * (n.y + 2)) * (n.x + 2);
    fluid.density[ighost] = fluid.density[ighostZm1];
    fluid.velocityX[ighost] = -fluid.velocityX[ighostZm1];
    fluid.velocityY[ighost] = -fluid.velocityY[ighostZm1];
    fluid.velocityZ[ighost] = -fluid.velocityZ[ighostZm1];
    fluid.momentumX[ighost] = -fluid.momentumX[ighostZm1];
    fluid.momentumY[ighost] = -fluid.momentumY[ighostZm1];
    fluid.momentumZ[ighost] = -fluid.momentumZ[ighostZm1];
  }

  void updateSimulationTime(real newTime) override {
    this->currentTime = newTime;
    this->bottomWallvx = amplitude * cos(2 * M_PI * freq * currentTime);
  }
};

using ICM = Hydro::ICM_Compressible_impl<ICMWalls>;

struct Parameters {
  real dt = 0.1;
  real3 boxSize = make_real3(32, 32, 32) * 100;
  int3 cellDim = {30, 30, 30};
  real bulkViscosity = 127.05;
  real speedOfSound = 14.67;
  real shearViscosity = 53.71;
  real temperature = 1;
  real initialDensity = 0.632;
  real relaxTime = 500;
  real simulationTime = -1;
  real printTime = 0;
  real wallFreq, wallAmplitude;
};

auto createICMIntegratorCompressible(std::shared_ptr<ParticleData> pd,
                                     Parameters ipar) {
  ICM::Parameters par;
  par.dt = ipar.dt;
  par.boxSize = ipar.boxSize;
  par.cellDim = ipar.cellDim;
  par.bulkViscosity = ipar.bulkViscosity;
  par.speedOfSound = ipar.speedOfSound;
  par.shearViscosity = ipar.shearViscosity;
  par.temperature = ipar.temperature;
  par.initialDensity = [=](real3 r) { return ipar.initialDensity; };
  par.walls = std::make_shared<ICMWalls>(ipar.wallAmplitude, ipar.wallFreq);
  return std::make_shared<ICM>(pd, par);
}

Parameters readParameters(std::string file) {
  InputFile in(file);
  Parameters par;
  in.getOption("dt", InputFile::Required) >> par.dt;
  in.getOption("boxSize", InputFile::Required) >> par.boxSize.x >>
      par.boxSize.y >> par.boxSize.z;
  in.getOption("cellDim", InputFile::Required) >> par.cellDim.x >>
      par.cellDim.y >> par.cellDim.z;
  in.getOption("bulkViscosity", InputFile::Required) >> par.bulkViscosity;
  in.getOption("shearViscosity", InputFile::Required) >> par.shearViscosity;
  in.getOption("speedOfSound", InputFile::Required) >> par.speedOfSound;
  in.getOption("temperature", InputFile::Required) >> par.temperature;
  in.getOption("initialDensity", InputFile::Required) >> par.initialDensity;
  in.getOption("relaxTime", InputFile::Optional) >> par.relaxTime;
  in.getOption("simulationTime", InputFile::Required) >> par.simulationTime;
  in.getOption("printTime", InputFile::Required) >> par.printTime;

  in.getOption("wallFreq", InputFile::Required) >> par.wallFreq;
  in.getOption("wallAmplitude", InputFile::Required) >> par.wallAmplitude;
  return par;
}

auto initializeParticles(Parameters par, std::shared_ptr<System> sys) {
  int numberParticles = 0;
  auto pd = std::make_shared<ParticleData>(numberParticles, sys);
  return pd;
}

template <class Iterator>
void write3DField(std::ofstream &out, Iterator vgpu, int size, int3 n,
                  Parameters par) {
  out << std::setprecision(8);
  std::vector<real> vx(size);
  thrust::copy(vgpu, vgpu + size, vx.begin());
  out << "#" << std::endl;
  for (int k = 0; k < n.z; k++) {
    int x = n.x / 2;
    int y = n.y / 2;
    int i = x + (y + k * n.y) * n.x;
    real z = par.boxSize.z * (k + 0.5) / n.z;
    out << z << " " << vx[i] << "\n";
  }
}

void writeFluidVelocity(std::shared_ptr<ICM> icm, Parameters par) {
  static std::ofstream out("vel.dat");
  auto vel = icm->getCurrentVelocity();
  auto n = icm->getGridSize();
  write3DField(out, vel.x(), vel.size(), n, par);
}

void writeFluidDensity(std::shared_ptr<ICM> icm, Parameters par) {
  static std::ofstream out("dens.dat");
  auto dens = icm->getCurrentDensity();
  auto n = icm->getGridSize();
  write3DField(out, dens.begin(), dens.size(), n, par);
}

void writeFluidFields(std::shared_ptr<ICM> integrator, Parameters simParams) {
  static std::ofstream out("fields.dat");
  int3 cellDim = integrator->getGridSize();
  auto velocity = integrator->getCurrentVelocity();
  auto density = integrator->getCurrentDensity();
  real3 L = simParams.boxSize;
  real3 cellSize = L / make_real3(cellDim);
  out << "#" << std::endl;
  out << std::setprecision(2 * sizeof(real));
  std::vector<real> vx(velocity.size()), vy(velocity.size()),
      vz(velocity.size()), rho(density.size());
  thrust::copy(velocity.x(), velocity.x() + velocity.size(), vx.begin());
  thrust::copy(velocity.y(), velocity.y() + velocity.size(), vy.begin());
  thrust::copy(velocity.z(), velocity.z() + velocity.size(), vz.begin());
  thrust::copy(density.begin(), density.end(), rho.begin());
  for (int k = 0; k < cellDim.z; ++k) {
    for (int j = 0; j < cellDim.y; ++j) {
      for (int i = 0; i < cellDim.x; ++i) {
        real x = -real(0.5) * L.x + i * (cellSize.x + real(0.5)) + cellSize.x;
        real y = -real(0.5) * L.y + j * (cellSize.y + real(0.5)) + cellSize.y;
        real z = -real(0.5) * L.z + k * (cellSize.z + real(0.5)) + cellSize.z;
        int ii = i + (j + k * cellDim.y) * cellDim.x;
        out << x << " " << y << " " << z << " " << vx[ii] << " " << vy[ii]
            << " " << vz[ii] << " 0 " << rho[ii] << "\n";
      }
    }
  }
  out << std::endl;
}

void writeFluidBottomGhostCellVelocity(std::shared_ptr<ICM> integrator,
                                       Parameters simParams) {
  static std::ofstream out("ghost.dat");
  int3 cellDim = integrator->getGridSize() + 2;
  auto velocity = integrator->getCurrentBottomGhostCellVelocity();
  real3 L = simParams.boxSize;
  real3 cellSize = L / make_real3(cellDim);
  out << "#" << std::endl;
  out << std::setprecision(2 * sizeof(real));
  std::vector<real> vx(velocity.size()), vy(velocity.size()),
      vz(velocity.size());
  thrust::copy(velocity.x(), velocity.x() + velocity.size(), vx.begin());
  thrust::copy(velocity.y(), velocity.y() + velocity.size(), vy.begin());
  thrust::copy(velocity.z(), velocity.z() + velocity.size(), vz.begin());
  for (int j = 0; j < cellDim.y; ++j) {
    for (int i = 0; i < cellDim.x; ++i) {
      real x = -real(0.5) * L.x + i * (cellSize.x - real(0.5)) + cellSize.x;
      real y = -real(0.5) * L.y + j * (cellSize.y - real(0.5)) + cellSize.y;
      real z = -real(0.5) * L.z;
      int ii = i + (j)*cellDim.x;
      out << x << " " << y << " " << z << " " << vx[ii] << " " << vy[ii] << " "
          << vz[ii] << "\n";
    }
  }
  out << std::endl;
}

int main(int argc, char *argv[]) {
  {
    auto sys = std::make_shared<System>(argc, argv);
    auto par = readParameters(argv[1]);
    auto pd = initializeParticles(par, sys);
    auto icm = createICMIntegratorCompressible(pd, par);
    {
      int relaxSteps = par.relaxTime / par.dt + 0.5;
      fori(0, relaxSteps) icm->forwardTime();
    }
    int ntimes = par.simulationTime / par.dt + 0.5;
    int sampleSteps = par.printTime / par.dt + 0.5;
    fori(0, ntimes + 1) {
      if (sampleSteps and i % sampleSteps == 0) {
        writeFluidVelocity(icm, par);
        writeFluidFields(icm, par);
        writeFluidBottomGhostCellVelocity(icm, par);
        // writeFluidDensity(icm, par);
      }
      icm->forwardTime();
    }
    System::log<System::MESSAGE>("Ending");
  }
  return 0;
}
