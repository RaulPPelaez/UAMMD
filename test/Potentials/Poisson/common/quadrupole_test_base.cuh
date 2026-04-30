#pragma once

#include "uammd.cuh"
#include "Interactor/DoublyPeriodic/DPPoissonSlab.cuh"
#include "Interactor/SpectralEwaldPoisson.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace uammd;

inline real4 calculateTheoreticalFieldPotential(real r, real d) {
    const real pi = M_PI;
    real4 field;
    field.x = 2.0/(4.0*pi)*(3*d*d/(r*r*r*r));
    field.y = 0.0;
    field.z = 0.0;
    field.w = 2.0/(4.0*pi)*(d*d/(r*r*r));
    return field;
}

class SimulationResult {
public:
    std::vector<real4> positions;
    std::vector<real4> fieldPotential;
    std::vector<real4> theoreticalFieldPotential;

    std::vector<real2> theoreticalError(real d){
        std::vector<real2> errors;
        for (size_t i = 0; i < positions.size(); ++i) {
            real4 theoretical = calculateTheoreticalFieldPotential(positions[i].x, d);
            theoreticalFieldPotential.push_back(theoretical);

            real errorE = sqrt(pow(fieldPotential[i].x - theoretical.x, 2.0));
            real errorV = sqrt(pow(fieldPotential[i].w - theoretical.w, 2.0));

            errors.push_back(make_real2(errorE, errorV));
        }
        return errors;
    }
};


inline real getL(const Poisson::Parameters& par) {
    return par.box.boxSize.x;
}

inline real getL(const DPPoissonSlab::Parameters& par) {
    return par.Lxy.x;
}

class QuadrupoleTest : public ::testing::Test {
protected:
    std::shared_ptr<System> sys;

    void SetUp() override { sys = std::make_shared<System>(); }
    void TearDown() override { sys->finish(); }

    template <typename PoissonType>
    SimulationResult runQuadrupoleSimulation(
        typename PoissonType::Parameters par, real d) {

        int N = 1000;
        auto pd = std::make_shared<ParticleData>(N, sys);

        real L = getL(par);

        {
            auto pos = pd->getPos(access::location::cpu, access::mode::write);
            auto charge = pd->getCharge(access::location::cpu, access::mode::write);

            pos[0] = make_real4(-d, 0, 0, 0);
            pos[1] = make_real4(0, 0, 0, 0);
            pos[2] = make_real4(0, 0, 0, 0);
            pos[3] = make_real4(d, 0, 0, 0);

            charge[0] = 1.0;
            charge[1] = -1.0;
            charge[2] = -1.0;
            charge[3] = 1.0;

            real r0 = 1.5;

            for (int i = 4; i < N; ++i) {
                pos[i] = make_real4(r0 + (i-4.0)/(N-4.0)*2.5, 0.0, 0.0, 0.0);
                charge[i] = 0;
            }
        }

        auto poisson = std::make_shared<PoissonType>(pd, par);
        auto field = poisson->computeFieldPotentialAtParticles();

        auto pos_h = pd->getPos(access::location::cpu, access::mode::read);

        SimulationResult result;
        for (int i = 4; i < N; ++i) {
            result.positions.push_back(pos_h[i]);
            result.fieldPotential.push_back(field[i]);
        }

        return result;
    }
};
