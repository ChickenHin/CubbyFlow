// This code is based on Jet framework.
// Copyright (c) 2018 Doyub Kim
// CubbyFlow is voxel-based fluid simulation engine for computer games.
// Copyright (c) 2020 CubbyFlow Team
// Core Part: Chris Ohk, Junwoo Hwang, Jihong Sin, Seungwoo Yoo
// AI Part: Dongheon Cho, Minseo Kim
// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "gtest/gtest.h"

#include <Core/Solver/Particle/MPM/MPMFluidSolver.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace CubbyFlow;

TEST(MPMFluidSolverManual, DamBreakReportsDensityErrorAndMomentum)
{
    constexpr double targetDensity = 1000.0;
    constexpr double particleSpacing = 0.05;
    constexpr double particleMass =
        targetDensity * particleSpacing * particleSpacing;
    MPMFluidSolver2 solver{ { 10, 10 },   { 0.1, 0.1 },  {},  0.02,
                            particleMass, targetDensity, 30.0 };
    solver.SetDragCoefficient(0.0);
    solver.SetTimeStepLimitScale(0.5);

    Array1<Vector2D> positions;
    for (size_t j = 0; j < 6; ++j)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            positions.Append(
                { 0.05 + particleSpacing * static_cast<double>(i),
                  0.05 + particleSpacing * static_cast<double>(j) });
        }
    }

    auto data = solver.GetMPMSystemData();
    data->AddParticles(positions);

    for (Frame frame{ 0, 1.0 / 240.0 }; frame.index < 40; ++frame)
    {
        solver.Update(frame);
    }

    const auto masses = data->ParticleMasses();
    const auto volumes = data->InitialVolumes();
    const auto volumeRatios = data->VolumeRatios();
    const auto velocities = data->Velocities();
    double maxDensityError = 0.0;
    Vector2D totalMomentum;

    for (size_t i = 0; i < data->NumberOfParticles(); ++i)
    {
        const double density = solver.GetConstitutiveModel().ComputeDensity(
            masses[i], volumes[i] * volumeRatios[i]);
        maxDensityError =
            std::max(maxDensityError, std::abs(density / targetDensity - 1.0));
        totalMomentum += masses[i] * velocities[i];

        EXPECT_TRUE(std::isfinite(density));
        EXPECT_TRUE(std::isfinite(velocities[i].x));
        EXPECT_TRUE(std::isfinite(velocities[i].y));
    }

    std::cout << "MPM dam-break max density error: " << 100.0 * maxDensityError
              << "%\n"
              << "MPM dam-break total momentum: (" << totalMomentum.x << ", "
              << totalMomentum.y << ")\n";

    EXPECT_TRUE(std::isfinite(maxDensityError));
    EXPECT_TRUE(std::isfinite(totalMomentum.x));
    EXPECT_TRUE(std::isfinite(totalMomentum.y));
}
