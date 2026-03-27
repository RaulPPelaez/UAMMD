#include<Interactor/DoublyPeriodic/DPPoissonSlab.cuh>
#include "uammd.cuh"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>


using namespace uammd;
using std::endl;
using std::make_shared;

class DoublyPeriodicPoissonTest : public ::testing::Test {
    protected:
        std::shared_ptr<System> sys;

        void SetUp() override {
            sys = make_shared<System>();
        }

        void TearDown() override {
            sys->finish();
        }

        auto createDPPoissonInteractor(std::shared_ptr<ParticleData> pd){
            DPPoissonSlab::Parameters par;
            par.Lxy = make_real2(185.762,185.762);
            par.H = 50; //Domain height
            DPPoissonSlab::Permitivity perm;
            perm.inside = 2.36e-2;
            perm.top = 2.36e-2;
            perm.bottom = 3.01e-4;
            par.permitivity = perm;
            par.gw = 0.25; //Width of the Gaussian sources
            par.Nxy = 72;
            auto poisson = make_shared<DPPoissonSlab>(pd, par);
            return poisson;
        }

        real4 computePeriodicField(){
            int N = 100;
            auto pd = make_shared<ParticleData>(N, sys);
            {
                auto pos = pd->getPos(access::location::cpu, access::mode::write);
                auto charge = pd->getCharge(access::location::cpu, access::mode::write);
                auto ori = make_real4(make_real3(sys->rng().uniform(-0.5, 0.5))*185, 0);
                pos[0] = ori;
                charge[0] = 1.0;
                for(int i = 1; i<N; ++i){
                    pos[i] = make_real4(make_real3(0.0,0.0,0.0), 0.0);
                    charge[i] = 0.0;
                }
            }
            auto poisson = createDPPoissonInteractor(pd);
            auto fieldPotential = poisson->computeFieldPotentialAtParticles();
            auto fieldPotential_h = fieldPotential;
            return fieldPotential_h[1];
        }
};

TEST_F(DoublyPeriodicPoissonTest, PoissonSlabTest) {
    real4 fieldPotential = computePeriodicField();
    EXPECT_TRUE(std::isfinite(fieldPotential.x));
    EXPECT_TRUE(std::isfinite(fieldPotential.y));
    EXPECT_TRUE(std::isfinite(fieldPotential.z));
    EXPECT_TRUE(std::isfinite(fieldPotential.w));
    }


