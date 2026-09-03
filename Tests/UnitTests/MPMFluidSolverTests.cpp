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

#include <Core/Emitter/PointParticleEmitter2.hpp>
#include <Core/Emitter/PointParticleEmitter3.hpp>
#include <Core/Geometry/Plane.hpp>
#include <Core/Geometry/RigidBodyCollider.hpp>
#include <Core/Solver/Particle/MPM/MPMFluidSolver.hpp>
#include <Core/Utils/Constants.hpp>
#include <Core/Utils/IterationUtils.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

using namespace CubbyFlow;

namespace
{
template <size_t N>
using VectorD = Vector<double, N>;

template <size_t N>
using VectorUZ = Vector<size_t, N>;

template <size_t N>
using PointEmitter =
    std::conditional_t<N == 2, PointParticleEmitter2, PointParticleEmitter3>;

template <size_t N>
class TestableMPMFluidSolver final : public MPMFluidSolver<N>
{
 public:
    using MPMFluidSolver<N>::MPMFluidSolver;

    void Initialize()
    {
        this->OnInitialize();
    }

    void BeginStep(double timeStepInSeconds)
    {
        this->OnBeginAdvanceTimeStep(timeStepInSeconds);
    }

    [[nodiscard]] unsigned int NumberOfSubTimeSteps(double interval) const
    {
        return this->GetNumberOfSubTimeSteps(interval);
    }
};

template <size_t N>
void UseOneFixedStep(MPMFluidSolver<N>* solver)
{
    solver->SetIsUsingFixedSubTimeSteps(true);
    solver->SetNumberOfFixedSubTimeSteps(1);
}

template <size_t N>
void ExpectParameters()
{
    MPMFluidSolver<N> solver;
    EXPECT_EQ(solver.GetClosedDomainBoundaryFlag(), DIRECTION_ALL);

    solver.SetClosedDomainBoundaryFlag(DIRECTION_LEFT | DIRECTION_UP);
    EXPECT_EQ(solver.GetClosedDomainBoundaryFlag(),
              DIRECTION_LEFT | DIRECTION_UP);
    EXPECT_DOUBLE_EQ(solver.GetTimeStepLimitScale(), 0.9);
    EXPECT_DOUBLE_EQ(solver.GetConstitutiveModel().GetTargetDensity(),
                     WATER_DENSITY);
    EXPECT_DOUBLE_EQ(solver.GetConstitutiveModel().GetSpeedOfSound(), 100.0);

    solver.SetTimeStepLimitScale(0.5);
    EXPECT_DOUBLE_EQ(solver.GetTimeStepLimitScale(), 0.5);
    EXPECT_THROW(solver.SetTimeStepLimitScale(0.0), std::invalid_argument);
    EXPECT_THROW(solver.SetTimeStepLimitScale(1.1), std::invalid_argument);
    EXPECT_THROW(
        solver.SetTimeStepLimitScale(std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
    EXPECT_THROW((MPMFluidSolver<N>{ VectorUZ<N>::MakeConstant(4),
                                     VectorD<N>::MakeConstant(1.0),
                                     {},
                                     -1.0,
                                     1.0 }),
                 std::invalid_argument);
    EXPECT_THROW((MPMFluidSolver<N>{ VectorUZ<N>::MakeConstant(4),
                                     VectorD<N>::MakeConstant(1.0),
                                     {},
                                     0.1,
                                     0.0 }),
                 std::invalid_argument);
}

template <size_t N>
void ExpectBuilder()
{
    const auto resolution = VectorUZ<N>::MakeConstant(7);
    const auto spacing = VectorD<N>::MakeConstant(0.25);
    const auto origin = VectorD<N>::MakeConstant(-1.0);
    auto built = MPMFluidSolver<N>::GetBuilder()
                     .WithResolution(resolution)
                     .WithGridSpacing(spacing)
                     .WithOrigin(origin)
                     .WithRadius(0.2)
                     .WithMass(3.0)
                     .WithTargetDensity(900.0)
                     .WithSpeedOfSound(20.0)
                     .WithEosExponent(5.0)
                     .WithNegativePressureScale(0.25)
                     .Build();

    const auto data = built.GetMPMSystemData();
    EXPECT_EQ(data->GridMass().Resolution(), resolution);
    EXPECT_EQ(data->GridMass().GridSpacing(), spacing);
    EXPECT_EQ(data->GridMass().Origin(), origin);
    EXPECT_DOUBLE_EQ(data->Radius(), 0.2);
    EXPECT_DOUBLE_EQ(data->Mass(), 3.0);
    EXPECT_DOUBLE_EQ(built.GetConstitutiveModel().GetTargetDensity(), 900.0);
    EXPECT_DOUBLE_EQ(built.GetConstitutiveModel().GetSpeedOfSound(), 20.0);
    EXPECT_DOUBLE_EQ(built.GetConstitutiveModel().GetEosExponent(), 5.0);
    EXPECT_DOUBLE_EQ(built.GetConstitutiveModel().GetNegativePressureScale(),
                     0.25);
    EXPECT_NE(MPMFluidSolver<N>::GetBuilder().MakeShared(), nullptr);
}

template <size_t N>
void ExpectEmitter()
{
    const auto resolution = VectorUZ<N>::MakeConstant(8);
    const auto spacing = VectorD<N>::MakeConstant(0.1);
    const auto position = spacing;
    VectorD<N> direction;
    direction[0] = 1.0;

    MPMFluidSolver<N> solver{ resolution, spacing };
    solver.SetGravity({});
    solver.SetDragCoefficient(0.0);
    auto emitter =
        std::make_shared<PointEmitter<N>>(position, direction, 0.0, 0.0, 1, 1);
    solver.SetEmitter(emitter);

    solver.Update(Frame{ 0, 0.001 });

    const auto data = solver.GetMPMSystemData();
    EXPECT_EQ(data->NumberOfParticles(), 1u);
    EXPECT_DOUBLE_EQ(data->InitialVolumes()[0], 1e-6);
}

template <size_t N>
void ExpectDefensiveChecks()
{
    const auto resolution = VectorUZ<N>::MakeConstant(8);
    const auto spacing = VectorD<N>::MakeConstant(1.0);
    const auto position = VectorD<N>::MakeConstant(3.5);

    TestableMPMFluidSolver<N> invalidVelocity{ resolution, spacing };
    auto invalidVelocityData = invalidVelocity.GetMPMSystemData();
    invalidVelocityData->AddParticle(position);
    invalidVelocity.Initialize();
    invalidVelocityData->Velocities()[0][0] =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(invalidVelocity.NumberOfSubTimeSteps(0.1)),
                 std::invalid_argument);

    invalidVelocityData->Velocities()[0][0] =
        std::numeric_limits<double>::max();
    EXPECT_THROW(static_cast<void>(invalidVelocity.NumberOfSubTimeSteps(0.1)),
                 std::invalid_argument);

    TestableMPMFluidSolver<N> invalidIncrement{ resolution, spacing };
    invalidIncrement.GetMPMSystemData()->AddParticle(position);
    invalidIncrement.Initialize();
    EXPECT_THROW(invalidIncrement.BeginStep(std::numeric_limits<double>::max()),
                 std::invalid_argument);
}

template <size_t N>
void ExpectReferenceVolumesAndAdaptiveSteps()
{
    const auto resolution = VectorUZ<N>::MakeConstant(8);
    const auto spacing = VectorD<N>::MakeConstant(0.1);

    TestableMPMFluidSolver<N> empty{ resolution, spacing };
    EXPECT_NO_THROW(empty.Initialize());
    EXPECT_EQ(empty.NumberOfSubTimeSteps(0.1), 1u);
    EXPECT_NO_THROW(empty.Update(Frame{ 0, 0.001 }));

    auto emittedData = empty.GetMPMSystemData();
    emittedData->AddParticle(VectorD<N>::MakeConstant(0.35));
    EXPECT_NO_THROW(empty.Update(Frame{ 1, 0.001 }));
    EXPECT_DOUBLE_EQ(emittedData->InitialVolumes()[0], 1e-6);

    TestableMPMFluidSolver<N> solver{ resolution, spacing, {},  0.01,
                                      2.0,        1000.0,  10.0 };
    const VectorD<N> position = VectorD<N>::MakeConstant(0.35);

    VectorD<N> velocity;
    velocity[0] = 3.0;

    auto data = solver.GetMPMSystemData();
    data->AddParticle(position, velocity);

    solver.Initialize();
    EXPECT_DOUBLE_EQ(data->InitialVolumes()[0], 0.002);
    EXPECT_EQ(solver.NumberOfSubTimeSteps(0.1), 15u);

    solver.SetTimeStepLimitScale(0.5);
    EXPECT_EQ(solver.NumberOfSubTimeSteps(0.1), 26u);

    data->InitialVolumes()[0] = 0.003;
    solver.BeginStep(0.001);
    EXPECT_DOUBLE_EQ(data->InitialVolumes()[0], 0.003);

    TestableMPMFluidSolver<N> fasterSound{ resolution, spacing, {},  0.01,
                                           2.0,        1000.0,  20.0 };
    fasterSound.GetMPMSystemData()->AddParticle(position, velocity);
    fasterSound.Initialize();
    EXPECT_EQ(fasterSound.NumberOfSubTimeSteps(0.1), 26u);

    TestableMPMFluidSolver<N> finerSpacing{
        resolution, VectorD<N>::MakeConstant(0.05), {}, 0.01, 2.0, 1000.0, 10.0
    };
    finerSpacing.GetMPMSystemData()->AddParticle(VectorD<N>::MakeConstant(0.2),
                                                 velocity);
    finerSpacing.Initialize();
    EXPECT_EQ(finerSpacing.NumberOfSubTimeSteps(0.1), 29u);

    TestableMPMFluidSolver<N> invalid{ resolution, spacing };
    invalid.GetMPMSystemData()->AddParticle(position);
    invalid.GetMPMSystemData()->InitialVolumes()[0] = -1.0;
    EXPECT_THROW(invalid.Initialize(), std::invalid_argument);
}

template <size_t N>
void ExpectUniformMotionAndExternalForces()
{
    const auto resolution = VectorUZ<N>::MakeConstant(8);
    const auto spacing = VectorD<N>::MakeConstant(1.0);
    const auto position = VectorD<N>::MakeConstant(3.5);

    MPMFluidSolver<N> uniform{ resolution, spacing };
    UseOneFixedStep(&uniform);
    uniform.SetGravity({});
    uniform.SetDragCoefficient(0.0);

    VectorD<N> velocity;
    velocity[0] = 0.2;

    auto uniformData = uniform.GetMPMSystemData();
    uniformData->AddParticle(position, velocity);
    uniform.Update(Frame{ 0, 0.001 });

    EXPECT_TRUE(uniformData->Velocities()[0].IsSimilar(velocity, 1e-12));
    EXPECT_TRUE(uniformData->Positions()[0].IsSimilar(
        position + 0.001 * velocity, 1e-12));

    MPMFluidSolver<N> gravitySolver{ resolution, spacing };
    UseOneFixedStep(&gravitySolver);
    gravitySolver.SetDragCoefficient(0.0);

    VectorD<N> gravity;
    gravity[1] = -2.0;
    gravitySolver.SetGravity(gravity);

    auto gravityData = gravitySolver.GetMPMSystemData();
    gravityData->AddParticle(position);
    gravitySolver.Update(Frame{ 0, 0.001 });

    EXPECT_TRUE(gravityData->Velocities()[0].IsSimilar(0.001 * gravity, 1e-12));

    MPMFluidSolver<N> dragSolver{ resolution, spacing, {}, 0.1, 2.0 };
    UseOneFixedStep(&dragSolver);
    dragSolver.SetGravity({});
    dragSolver.SetDragCoefficient(4.0);

    velocity = {};
    velocity[0] = 1.0;

    auto dragData = dragSolver.GetMPMSystemData();
    dragData->AddParticle(position, velocity);
    dragSolver.Update(Frame{ 0, 0.001 });

    velocity[0] = 0.998;
    EXPECT_TRUE(dragData->Velocities()[0].IsSimilar(velocity, 1e-12));
}

template <size_t N>
void ExpectCompressedParticlesMoveOutwardAndStayFinite()
{
    const auto resolution = VectorUZ<N>::MakeConstant(8);
    const auto spacing = VectorD<N>::MakeConstant(1.0);
    MPMFluidSolver<N> solver{ resolution, spacing, {}, 0.1, 1.0, 1000.0, 10.0 };

    UseOneFixedStep(&solver);
    solver.SetGravity({});
    solver.SetDragCoefficient(0.0);

    VectorD<N> left = VectorD<N>::MakeConstant(3.5);
    VectorD<N> right = left;
    left[0] = 3.25;
    right[0] = 3.75;

    auto data = solver.GetMPMSystemData();
    data->AddParticle(left);
    data->AddParticle(right);
    data->InitialVolumes()[0] = 0.001;
    data->InitialVolumes()[1] = 0.001;
    data->VolumeRatios()[0] = 0.5;
    data->VolumeRatios()[1] = 0.5;

    for (int frame = 0; frame < 4; ++frame)
    {
        solver.Update(Frame{ frame, 1e-5 });
    }

    EXPECT_LT(data->Velocities()[0][0], 0.0);
    EXPECT_GT(data->Velocities()[1][0], 0.0);

    for (size_t i = 0; i < data->NumberOfParticles(); ++i)
    {
        EXPECT_GT(data->VolumeRatios()[i], 0.0);
        EXPECT_TRUE(std::isfinite(data->VolumeRatios()[i]));

        for (double component : data->Positions()[i])
        {
            EXPECT_TRUE(std::isfinite(component));
        }
        for (double component : data->Velocities()[i])
        {
            EXPECT_TRUE(std::isfinite(component));
        }
    }
}

template <size_t N>
void ExpectClosedDomainWalls()
{
    constexpr std::array lowerFlags{ DIRECTION_LEFT, DIRECTION_DOWN,
                                     DIRECTION_BACK };
    constexpr std::array upperFlags{ DIRECTION_RIGHT, DIRECTION_UP,
                                     DIRECTION_FRONT };
    const auto resolution = VectorUZ<N>::MakeConstant(4);
    const auto spacing = VectorD<N>::MakeConstant(1.0);

    for (size_t axis = 0; axis < N; ++axis)
    {
        for (bool isUpper : { false, true })
        {
            TestableMPMFluidSolver<N> solver{ resolution, spacing };
            solver.SetClosedDomainBoundaryFlag(isUpper ? upperFlags[axis]
                                                       : lowerFlags[axis]);
            solver.SetGravity({});
            solver.SetDragCoefficient(0.0);

            VectorD<N> position = VectorD<N>::MakeConstant(2.0);
            VectorD<N> velocity;

            position[axis] = isUpper ? 3.75 : 0.25;
            velocity[axis] = isUpper ? 1.0 : -1.0;

            auto data = solver.GetMPMSystemData();
            data->AddParticle(position, velocity);

            solver.BeginStep(1e-3);

            VectorUZ<N> nodeIndex = VectorUZ<N>::MakeConstant(2);
            nodeIndex[axis] =
                isUpper ? data->GridMass().DataSize()[axis] - 1 : 0;

            ASSERT_GT(data->GridMass()(nodeIndex), 0.0);
            EXPECT_DOUBLE_EQ(data->GridVelocities()(nodeIndex)[axis], 0.0);
        }
    }

    TestableMPMFluidSolver<N> open{ resolution, spacing };
    open.SetClosedDomainBoundaryFlag(DIRECTION_NONE);
    open.SetGravity({});
    open.SetDragCoefficient(0.0);

    VectorD<N> position = VectorD<N>::MakeConstant(2.0);
    VectorD<N> velocity;

    position[0] = 0.25;
    velocity[0] = -1.0;

    auto data = open.GetMPMSystemData();
    data->AddParticle(position, velocity);

    open.BeginStep(1e-3);

    VectorUZ<N> nodeIndex = VectorUZ<N>::MakeConstant(2);
    nodeIndex[0] = 0;

    ASSERT_GT(data->GridMass()(nodeIndex), 0.0);
    EXPECT_NEAR(data->GridVelocities()(nodeIndex)[0], -1.0, 1e-12);
}

template <size_t N>
void ExpectMovingColliderAffectsGrid()
{
    TestableMPMFluidSolver<N> solver{ VectorUZ<N>::MakeConstant(4),
                                      VectorD<N>::MakeConstant(1.0) };
    solver.SetClosedDomainBoundaryFlag(DIRECTION_NONE);
    solver.SetGravity({});
    solver.SetDragCoefficient(0.0);

    VectorD<N> normal;
    normal[0] = 1.0;

    auto collider = std::make_shared<RigidBodyCollider<N>>(
        std::make_shared<Plane<N>>(normal, VectorD<N>{}));
    collider->linearVelocity[0] = 1.0;

    solver.SetCollider(collider);

    VectorD<N> position = VectorD<N>::MakeConstant(2.0);
    position[0] = 0.25;

    auto data = solver.GetMPMSystemData();
    data->AddParticle(position);

    solver.BeginStep(1e-3);

    VectorUZ<N> nodeIndex = VectorUZ<N>::MakeConstant(2);
    nodeIndex[0] = 0;

    ASSERT_GT(data->GridMass()(nodeIndex), 0.0);
    EXPECT_DOUBLE_EQ(data->GridVelocities()(nodeIndex)[0], 1.0);
}

template <size_t N>
void ExpectFrictionAffectsGrid()
{
    const auto runCase = [](double frictionCoefficient) {
        TestableMPMFluidSolver<N> solver{ VectorUZ<N>::MakeConstant(4),
                                          VectorD<N>::MakeConstant(1.0) };
        solver.SetClosedDomainBoundaryFlag(DIRECTION_NONE);
        solver.SetGravity({});
        solver.SetDragCoefficient(0.0);

        VectorD<N> normal;
        normal[1] = 1.0;

        VectorD<N> point;
        point[1] = 2.0;

        auto collider = std::make_shared<RigidBodyCollider<N>>(
            std::make_shared<Plane<N>>(normal, point));
        collider->SetFrictionCoefficient(frictionCoefficient);

        solver.SetCollider(collider);

        VectorD<N> position = VectorD<N>::MakeConstant(2.25);
        VectorD<N> velocity;

        velocity[0] = 1.0;
        velocity[1] = -1.0;

        auto data = solver.GetMPMSystemData();
        data->AddParticle(position, velocity);

        solver.BeginStep(1e-3);

        const VectorUZ<N> nodeIndex = VectorUZ<N>::MakeConstant(2);
        EXPECT_GT(data->GridMass()(nodeIndex), 0.0);

        return data->GridVelocities()(nodeIndex);
    };

    const auto frictionless = runCase(0.0);
    EXPECT_NEAR(frictionless[0], 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(frictionless[1], 0.0);

    const auto frictional = runCase(1.0);
    EXPECT_DOUBLE_EQ(frictional[0], 0.0);
    EXPECT_DOUBLE_EQ(frictional[1], 0.0);
}

template <size_t N>
void ExpectParticleDomainProjection()
{
    const auto runCase = [](int boundaryFlag) {
        MPMFluidSolver<N> solver{ VectorUZ<N>::MakeConstant(4),
                                  VectorD<N>::MakeConstant(1.0) };

        UseOneFixedStep(&solver);

        solver.SetClosedDomainBoundaryFlag(boundaryFlag);
        solver.SetGravity({});
        solver.SetDragCoefficient(0.0);

        VectorD<N> position = VectorD<N>::MakeConstant(2.0);
        VectorD<N> velocity;

        position[0] = -0.01;
        velocity[0] = -1.0;

        auto data = solver.GetMPMSystemData();
        data->AddParticle(position, velocity);

        solver.Update(Frame{ 0, 1e-3 });

        return std::array{ data->Positions()[0][0], data->Velocities()[0][0] };
    };

    const auto closed = runCase(DIRECTION_LEFT);
    EXPECT_DOUBLE_EQ(closed[0], 0.0);
    EXPECT_GE(closed[1], 0.0);

    const auto open = runCase(DIRECTION_NONE);
    EXPECT_NEAR(open[0], -0.011, 1e-12);
    EXPECT_NEAR(open[1], -1.0, 1e-12);
}

template <size_t N>
void ExpectParticleColliderProjection()
{
    constexpr double radius = 0.1;
    MPMFluidSolver<N> solver{ VectorUZ<N>::MakeConstant(4),
                              VectorD<N>::MakeConstant(1.0),
                              {},
                              radius,
                              1.0 };

    UseOneFixedStep(&solver);

    solver.SetClosedDomainBoundaryFlag(DIRECTION_NONE);
    solver.SetGravity({});
    solver.SetDragCoefficient(0.0);

    VectorD<N> normal;
    normal[1] = 1.0;

    VectorD<N> point;
    point[1] = 1.0;

    solver.SetCollider(std::make_shared<RigidBodyCollider<N>>(
        std::make_shared<Plane<N>>(normal, point)));

    VectorD<N> position = VectorD<N>::MakeConstant(2.0);
    VectorD<N> velocity;

    position[1] = 0.95;
    velocity[1] = -1.0;

    auto data = solver.GetMPMSystemData();
    data->AddParticle(position, velocity);

    solver.Update(Frame{ 0, 1e-3 });

    EXPECT_GE(data->Positions()[0][1], point[1] + radius);
    EXPECT_GE(data->Velocities()[0][1], 0.0);
}

template <size_t N>
double ParticleMeasure(double spacing)
{
    double result = 1.0;

    for (size_t axis = 0; axis < N; ++axis)
    {
        result *= spacing;
    }

    return result;
}

template <size_t N>
void AddParticleBlock(MPMFluidSolver<N>* solver, const VectorUZ<N>& size,
                      double spacing)
{
    Array1<VectorD<N>> positions;

    ForEachIndex(size, [&positions, spacing](auto... rawIndices) {
        const VectorUZ<N> index{ rawIndices... };
        VectorD<N> position;

        for (size_t axis = 0; axis < N; ++axis)
        {
            position[axis] = 0.05 + spacing * static_cast<double>(index[axis]);
        }

        positions.Append(position);
    });

    solver->GetMPMSystemData()->AddParticles(positions);
}

template <size_t N>
void ExpectHydrostaticColumnRemainsBounded()
{
    constexpr double particleSpacing = 0.1;
    constexpr double targetDensity = 1000.0;
    const double particleMass =
        targetDensity * ParticleMeasure<N>(particleSpacing);

    MPMFluidSolver<N> solver{ VectorUZ<N>::MakeConstant(6),
                              VectorD<N>::MakeConstant(0.1),
                              {},
                              0.02,
                              particleMass,
                              targetDensity,
                              50.0 };
    solver.SetDragCoefficient(0.0);
    solver.SetTimeStepLimitScale(0.5);

    VectorUZ<N> blockSize = VectorUZ<N>::MakeConstant(6);
    blockSize[1] = 4;

    AddParticleBlock(&solver, blockSize, particleSpacing);

    for (int frame = 0; frame < 24; ++frame)
    {
        solver.Update(Frame{ frame, 1.0 / 240.0 });
    }

    const auto data = solver.GetMPMSystemData();
    const auto masses = data->ParticleMasses();
    const auto initialVolumes = data->InitialVolumes();
    const auto volumeRatios = data->VolumeRatios();
    double maxDensityError = 0.0;

    for (size_t i = 0; i < data->NumberOfParticles(); ++i)
    {
        const double density = solver.GetConstitutiveModel().ComputeDensity(
            masses[i], initialVolumes[i] * volumeRatios[i]);
        const double pressure =
            solver.GetConstitutiveModel().ComputePressure(density);

        EXPECT_TRUE(std::isfinite(density));
        EXPECT_TRUE(std::isfinite(pressure));
        EXPECT_GT(density, 0.0);

        maxDensityError =
            std::max(maxDensityError, std::abs(density / targetDensity - 1.0));
    }

    EXPECT_LT(maxDensityError, 0.03);
}

template <size_t N>
void ExpectDamBreakConservesMassAndStaysInDomain()
{
    constexpr double particleSpacing = 0.05;
    constexpr double targetDensity = 1000.0;
    const double particleMass =
        targetDensity * ParticleMeasure<N>(particleSpacing);

    MPMFluidSolver<N> solver{ VectorUZ<N>::MakeConstant(10),
                              VectorD<N>::MakeConstant(0.1),
                              {},
                              0.02,
                              particleMass,
                              targetDensity,
                              30.0 };
    solver.SetDragCoefficient(0.0);
    solver.SetTimeStepLimitScale(0.5);

    VectorUZ<N> blockSize = VectorUZ<N>::MakeConstant(3);
    blockSize[1] = 6;

    AddParticleBlock(&solver, blockSize, particleSpacing);

    const auto data = solver.GetMPMSystemData();
    const size_t initialCount = data->NumberOfParticles();
    const double initialMass = particleMass * static_cast<double>(initialCount);
    double initialMaxX = 0.0;

    for (const auto& position : data->Positions())
    {
        initialMaxX = std::max(initialMaxX, position[0]);
    }

    for (int frame = 0; frame < 40; ++frame)
    {
        solver.Update(Frame{ frame, 1.0 / 240.0 });
    }

    const auto domain = data->GridMass().GetBoundingBox();
    double finalMass = 0.0;
    double finalMaxX = 0.0;

    for (size_t i = 0; i < data->NumberOfParticles(); ++i)
    {
        finalMass += data->ParticleMasses()[i];
        finalMaxX = std::max(finalMaxX, data->Positions()[i][0]);

        for (size_t axis = 0; axis < N; ++axis)
        {
            EXPECT_GE(data->Positions()[i][axis], domain.lowerCorner[axis]);
            EXPECT_LE(data->Positions()[i][axis], domain.upperCorner[axis]);
        }

        EXPECT_TRUE(std::isfinite(data->VolumeRatios()[i]));
        EXPECT_GT(data->VolumeRatios()[i], 0.0);
    }

    EXPECT_EQ(data->NumberOfParticles(), initialCount);
    EXPECT_NEAR(finalMass, initialMass,
                std::numeric_limits<double>::epsilon() * initialMass);
    EXPECT_GT(finalMaxX, initialMaxX + 0.01);
}
}  // namespace

TEST(MPMFluidSolver, ParametersAndBuilder)
{
    ExpectParameters<2>();
    ExpectParameters<3>();
    ExpectBuilder<2>();
    ExpectBuilder<3>();
}

TEST(MPMFluidSolver, Emitter)
{
    ExpectEmitter<2>();
    ExpectEmitter<3>();
}

TEST(MPMFluidSolver, DefensiveChecks)
{
    ExpectDefensiveChecks<2>();
    ExpectDefensiveChecks<3>();
}

TEST(MPMFluidSolver, ReferenceVolumesAndAdaptiveSteps)
{
    ExpectReferenceVolumesAndAdaptiveSteps<2>();
    ExpectReferenceVolumesAndAdaptiveSteps<3>();
}

TEST(MPMFluidSolver, UniformMotionAndExternalForces)
{
    ExpectUniformMotionAndExternalForces<2>();
    ExpectUniformMotionAndExternalForces<3>();
}

TEST(MPMFluidSolver, CompressedParticlesMoveOutwardAndStayFinite)
{
    ExpectCompressedParticlesMoveOutwardAndStayFinite<2>();
    ExpectCompressedParticlesMoveOutwardAndStayFinite<3>();
}

TEST(MPMFluidSolver, ClosedDomainWalls)
{
    ExpectClosedDomainWalls<2>();
    ExpectClosedDomainWalls<3>();
}

TEST(MPMFluidSolver, ColliderContact)
{
    ExpectMovingColliderAffectsGrid<2>();
    ExpectMovingColliderAffectsGrid<3>();
    ExpectFrictionAffectsGrid<2>();
    ExpectFrictionAffectsGrid<3>();
}

TEST(MPMFluidSolver, ParticleDomainProjection)
{
    ExpectParticleDomainProjection<2>();
    ExpectParticleDomainProjection<3>();
}

TEST(MPMFluidSolver, ParticleColliderProjection)
{
    ExpectParticleColliderProjection<2>();
    ExpectParticleColliderProjection<3>();
}

TEST(MPMFluidSolver, HydrostaticColumnRemainsBounded)
{
    ExpectHydrostaticColumnRemainsBounded<2>();
    ExpectHydrostaticColumnRemainsBounded<3>();
}

TEST(MPMFluidSolver, DamBreakConservesMassAndStaysInDomain)
{
    ExpectDamBreakConservesMassAndStaysInDomain<2>();
    ExpectDamBreakConservesMassAndStaysInDomain<3>();
}
