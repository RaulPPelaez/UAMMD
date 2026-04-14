#include "Interactor/SpectralEwaldPoisson.cuh"
#include "Interactor/DoublyPeriodic/DPPoissonSlab.cuh"
#include "uammd.cuh"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace uammd;
using std::endl;
using std::make_shared;

// Helper function
real4 calculateTheoreticalFieldPotential(real r, real d) {
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
    std::vector<real4> positions = std::vector<real4>();
    std::vector<real4> fieldPotential = std::vector<real4>();
    std::vector<real4> theoreticalFieldPotential = std::vector<real4>();

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

// Helper functions (overloads instead of template specialization)
real getL(const Poisson::Parameters& par) {
    return par.box.boxSize.x;
}

real getL(const DPPoissonSlab::Parameters& par) {
    return par.Lxy.x;
}

class QuadrupoleTest : public ::testing::Test {
protected:
    std::shared_ptr<System> sys;

    void SetUp() override { sys = make_shared<System>(); }
    void TearDown() override { sys->finish(); }

    template <typename PoissonType>
    SimulationResult runQuadrupoleSimulation(typename PoissonType::Parameters par, real d) {
        int N = 1000;
        auto pd = make_shared<ParticleData>(N, sys);

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
                pos[i] = make_real4(r0 + (i-4.0)/(N-4.0) * 2.5, 0.0, 0.0, 0.0); // Spread particles from r0 to r0+2.5 (1.5 to 4.0)
                charge[i] = 0;
            }
        }

        auto poisson = make_shared<PoissonType>(pd, par);
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

TEST_F(QuadrupoleTest, SpectralEwaldPoisson) {
    real testTolerance = 1e-8;

    Poisson::Parameters par;
    par.box = Box(make_real3(50.0, 50.0, 50.0));
    par.gw = 0.2*sqrt(2.0/3.0);
    par.tolerance = 1e-14;
    par.epsilon = 1.0;
    par.split = 1.53093;
    real d = 0.01;

    auto result = runQuadrupoleSimulation<Poisson>(par, d);
    auto errors = result.theoreticalError(d);

    for (size_t i = 0; i < errors.size(); ++i) {

        EXPECT_LT(errors[i].x, testTolerance);
        EXPECT_LT(errors[i].y, testTolerance*10);
        std::cout
            << result.positions[i].x << " "
            << result.fieldPotential[i].x << " "
            << result.fieldPotential[i].w << " "
            << result.theoreticalFieldPotential[i].x << " "
            << result.theoreticalFieldPotential[i].w << " "
            << errors[i].x << " "
            << errors[i].y
            << endl;
    }
}

TEST_F(QuadrupoleTest, DPPoissonSlab) {
    real testTolerance = 1e-5;

    DPPoissonSlab::Parameters par;
    par.Lxy = make_real2(50.0, 50.0);
    par.H = 5.0;
    par.gw = 0.2*sqrt(2.0/3.0);

    DPPoissonSlab::Permitivity perm;
    perm.inside = 1.0;
    perm.top = 1.0;
    perm.bottom = 1.0;
    par.permitivity = perm;
    par.split = 1.70103;
    real d = 0.08;
    auto result = runQuadrupoleSimulation<DPPoissonSlab>(par, d);
    auto errors = result.theoreticalError(d);

    for (size_t i = 0; i < errors.size(); ++i) {
        std::cout
            << result.fieldPotential[i].x << " "
            << result.fieldPotential[i].w << " "
            << result.positions[i].x << " "
            << errors[i]
            << endl;
        EXPECT_LT(errors[i].x, testTolerance);
        EXPECT_LT(errors[i].y, testTolerance*10);
    }
}
