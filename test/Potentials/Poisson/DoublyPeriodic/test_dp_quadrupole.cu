#include "quadrupole_test_base.cuh"
#include "Interactor/DoublyPeriodic/DPPoissonSlab.cuh"

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
        EXPECT_LT(errors[i].x, testTolerance);
        EXPECT_LT(errors[i].y, testTolerance*10);
    }
}
