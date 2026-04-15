#include "quadrupole_test_base.cuh"
#include "Interactor/SpectralEwaldPoisson.cuh"

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
    }
}
