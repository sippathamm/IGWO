//
// Created by Sippawit Thammawiset on 29/1/2024 AD.
//

#ifndef BENCHMARK_FUNCTION_H
#define BENCHMARK_FUNCTION_H

#include <random>

namespace Benchmark
{
    typedef int FUNCTION_NAME;

    enum
    {
        SPHERE = 0,
        SCHWEFEL_S_2_22 = 1,
        SCHWEFEL_S_1_20 = 2,
        ROSENBROCK = 3,
        STEP = 4,
        QUARTIC_NOISE = 5,
        SCHWEFEL_S_2_26 = 6,
        RASTRIGIN = 7,
        ACKLEY = 8,
        GRIEWANK = 9
    };

    double GenerateRandom (double LowerBound = 0.0, double UpperBound = 1.0)
    {
        std::random_device Engine;
        std::uniform_real_distribution<double> RandomDistribution(0.0, 1.0);
        return LowerBound + RandomDistribution(Engine) * (UpperBound - LowerBound);
    }

    namespace Function
    {
        double Sphere (const std::vector<double> &Position)
        {
            double Sum = 0.0;

            for (const double &i : Position)
            {
                Sum += i * i;
            }

            return Sum;
        }

        double Schwefel_s_2_22 (const std::vector<double> &Position)
        {
            double Term1 = 0.0;
            double Term2 = 1.0;

            for (const double &i : Position)
            {
                Term1 += std::abs(i);
                Term2 *= std::abs(i);
            }

            return Term1 + Term2;
        }

        double Schwefel_s_1_20 (const std::vector<double> &Position)
        {
            double Sum = 0.0;

            for (int i = 0; i < Position.size(); ++i)
            {
                double InnerSum = 0.0;

                for (int j = 0; j <= i; ++j)
                {
                    InnerSum += Position[j];
                }

                Sum += InnerSum * InnerSum;
            }

            return Sum;
        }

        double Rosenbrock (const std::vector<double> &Position)
        {
            double Sum = 0.0;

            for (int i = 0; i < Position.size() - 1; ++i)
            {
                double Term1 = 100 * (Position[i + 1] - Position[i] * Position[i]) * (Position[i + 1] - Position[i] * Position[i]);
                double Term2 = (Position[i] - 1) * (Position[i] - 1);

                Sum += Term1 + Term2;
            }

            return Sum;
        }

        double Step (const std::vector<double> &Position)
        {
            double Sum = 0.0;

            for (const double &i : Position)
            {
                Sum += (i + 0.5) * (i + 0.5);
            }

            return Sum;
        }

        double QuarticNoise (const std::vector<double> &Position)
        {
            double Sum = 0.0f;

            for (int i = 0; i < Position.size(); ++i)
            {
                Sum += i * Position[i] * Position[i] * Position[i] * Position[i];
            }

            return Sum + GenerateRandom(0.0, 1.0);
        }

        double Schwefel_s_2_26 (const std::vector<double> &Position)
        {
            double Sum = 0.0;

            for (const double &i : Position)
            {
                Sum += i * sin(sqrt(abs(i)));
            }

            return -Sum;
        }

        double Rastrigin (const std::vector<double> &Position)
        {
            double Sum = 0.0;

            for (const double &i : Position)
            {
                Sum += i * i - 10 * cos(2.0 * M_PI * i) + 10.0;
            }

            return Sum;
        }

        double Ackley (const std::vector<double> &Position)
        {
            int Dimension = static_cast<int>(Position.size());

            double SumSquare = 0.0;
            double SumCosine = 0.0;

            for (const double &i : Position)
            {
                SumSquare += i * i;
                SumCosine += cos(2.0 * M_PI * i);
            }

            double Term1 = -20.0 * exp(-0.2 * sqrt(SumSquare / Dimension));
            double Term2 = -exp(SumCosine / Dimension);

            return Term1 + Term2 + 20.0 + expf(1.0);
        }

        double Griewank (const std::vector<double> &Position)
        {
            double SumSquare = 0.0;
            double ProductCosine = 1.0;

            for (int i = 0; i < Position.size(); i++)
            {
                SumSquare += Position[i] * Position[i];
                ProductCosine *= cos(Position[i] / sqrt(i + 1));
            }

            return SumSquare / 4000 - ProductCosine + 1;
        }
    } // Benchmark::Function

    namespace Property
    {
        void Sphere (int &NVariable, std::vector<double> &LowerBound, std::vector<double> &UpperBound)
        {
            NVariable = 30;

            LowerBound = std::vector<double> (NVariable, -100);
            UpperBound = std::vector<double> (NVariable, 100);
        }

        void Schwefel_s_2_22 (int &NVariable, std::vector<double> &LowerBound, std::vector<double> &UpperBound)
        {
            NVariable = 30;

            LowerBound = std::vector<double> (NVariable, -10);
            UpperBound = std::vector<double> (NVariable, 10);
        }

        void Schwefel_s_1_20 (int &NVariable, std::vector<double> &LowerBound, std::vector<double> &UpperBound)
        {
            NVariable = 30;

            LowerBound = std::vector<double> (NVariable, -100);
            UpperBound = std::vector<double> (NVariable, 100);
        }

        void Rosenbrock (int &NVariable, std::vector<double> &LowerBound, std::vector<double> &UpperBound)
        {
            NVariable = 30;

            LowerBound = std::vector<double> (NVariable, -30);
            UpperBound = std::vector<double> (NVariable, 30);
        }

        void Step (int &NVariable, std::vector<double> &LowerBound, std::vector<double> &UpperBound)
        {
            NVariable = 30;

            LowerBound = std::vector<double> (NVariable, -100);
            UpperBound = std::vector<double> (NVariable, 100);
        }

        void QuarticNoise (int &NVariable, std::vector<double> &LowerBound, std::vector<double> &UpperBound)
        {
            NVariable = 30;

            LowerBound = std::vector<double> (NVariable, -1.28);
            UpperBound = std::vector<double> (NVariable, 1.28);
        }

        void Schwefel_s_2_26 (int &NVariable, std::vector<double> &LowerBound, std::vector<double> &UpperBound)
        {
            NVariable = 30;

            LowerBound = std::vector<double> (NVariable, -500);
            UpperBound = std::vector<double> (NVariable, 500);
        }

        void Rastrigin (int &NVariable, std::vector<double> &LowerBound, std::vector<double> &UpperBound)
        {
            NVariable = 30;

            LowerBound = std::vector<double> (NVariable, -5.12);
            UpperBound = std::vector<double> (NVariable, 5.12);
        }

        void Ackley (int &NVariable, std::vector<double> &LowerBound, std::vector<double> &UpperBound)
        {
            NVariable = 30;

            LowerBound = std::vector<double> (NVariable, -32);
            UpperBound = std::vector<double> (NVariable, 32);
        }

        void Griewank (int &NVariable, std::vector<double> &LowerBound, std::vector<double> &UpperBound)
        {
            NVariable = 30;

            LowerBound = std::vector<double> (NVariable, -600);
            UpperBound = std::vector<double> (NVariable, 600);
        }
    } // Benchmark::Property

    double BenchmarkFunction (FUNCTION_NAME FUNCTION, const std::vector<double> &Position)
    {
        switch (FUNCTION)
        {
            case SPHERE:
                return Function::Sphere(Position);
            case SCHWEFEL_S_2_22:
                return Function::Schwefel_s_2_22(Position);
            case SCHWEFEL_S_1_20:
                return Function::Schwefel_s_1_20(Position);
            case ROSENBROCK:
                return Function::Rosenbrock(Position);
            case STEP:
                return Function::Step(Position);
            case QUARTIC_NOISE:
                return Function::QuarticNoise(Position);
            case SCHWEFEL_S_2_26:
                return Function::Schwefel_s_2_26(Position);
            case RASTRIGIN:
                return Function::Rastrigin(Position);
            case ACKLEY:
                return Function::Ackley(Position);
            case GRIEWANK:
                return Function::Griewank(Position);

            default:
                return Function::Sphere(Position);
        }

        return -1;
    }

    void BenchmarkProperty (FUNCTION_NAME FUNCTION,
                            int &NVariable,
                            std::vector<double> &LowerBound, std::vector<double> &UpperBound)
    {
        switch (FUNCTION)
        {
            case SPHERE:
                Property::Sphere(NVariable, LowerBound, UpperBound);
                break;
            case SCHWEFEL_S_2_22:
                Property::Schwefel_s_2_22(NVariable, LowerBound, UpperBound);
                break;
            case SCHWEFEL_S_1_20:
                Property::Schwefel_s_1_20(NVariable, LowerBound, UpperBound);
                break;
            case ROSENBROCK:
                Property::Rosenbrock(NVariable, LowerBound, UpperBound);
                break;
            case STEP:
                Property::Step(NVariable, LowerBound, UpperBound);
                break;
            case QUARTIC_NOISE:
                Property::QuarticNoise(NVariable, LowerBound, UpperBound);
                break;
            case SCHWEFEL_S_2_26:
                Property::Schwefel_s_2_26(NVariable, LowerBound, UpperBound);
                break;
            case RASTRIGIN:
                Property::Rastrigin(NVariable, LowerBound, UpperBound);
                break;
            case ACKLEY:
                Property::Ackley(NVariable, LowerBound, UpperBound);
                break;
            case GRIEWANK:
                Property::Griewank(NVariable, LowerBound, UpperBound);
                break;

            default:
                Property::Sphere(NVariable, LowerBound, UpperBound);
        }
    }
}

#endif //BENCHMARK_FUNCTION_H
