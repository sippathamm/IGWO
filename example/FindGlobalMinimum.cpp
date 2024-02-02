//
// Created by Sippawit Thammawiset on 25/1/2024 AD.
//

#include <iostream>

#include "GWO.h"
#include "BenchmarkFunction.h"

double GetMean (const std::vector<double> &Sample);
double GetVariance (const std::vector<double> &Sample);

double FitnessFunction (const std::vector<double> &Position)
{
    // Define your fitness function here

    return Benchmark::BenchmarkFunction(Benchmark::SCHWEFEL_S_2_26, Position);
}

int main() {
    std::vector<double> LowerBound, UpperBound;

    int MaximumIteration, NPopulation, NVariable;
    double Theta = 2.0f, K = 1.5f;
    double Maximum_a = 2.2f, Minimum_a = 0.02f;
    double MaximumWeight = 0.9f, MinimumWeight = 0.4f;
    double VelocityFactor = 0.5f;

    Benchmark::BenchmarkCondition(Benchmark::SCHWEFEL_S_2_26,
                                  LowerBound, UpperBound,
                                  MaximumIteration, NPopulation, NVariable);

    int NRun = 30;

    // Save the results
    double Maximum = -INFINITY;
    double Minimum = INFINITY;
    std::vector<double> Sample;

    for (int Run = 1; Run <= NRun; Run++)
    {
        std::cout << "Run:\t" << Run << std::endl;

        Optimizer::AGWO GWO(LowerBound, UpperBound,
                            MaximumIteration, NPopulation, NVariable,
                            Theta, K,
                            Maximum_a, Minimum_a,
                            MaximumWeight, MinimumWeight,
                            VelocityFactor,
                            false);

        GWO.SetFitnessFunction(FitnessFunction);

        if (GWO.Run())
        {
            auto GlobalBestPosition = GWO.GetGlobalBestPosition();

            std::cout << "Global Best Position:\t";
            for (const auto &i : GlobalBestPosition)
            {
                std::cout << i << "\t";
            }
            std::cout << std::endl;

            double GlobalBestFitnessValue = GWO.GetGlobalBestFitnessValue();

            std::cout << "Global Best Fitness Value:\t" << GlobalBestFitnessValue << std::endl;

            Maximum = std::max(Maximum, GlobalBestFitnessValue);
            Minimum = std::min(Minimum, GlobalBestFitnessValue);

            Sample.push_back(GlobalBestFitnessValue);
        }
        else
        {
            break;
        }

        std::cout << "------------------------" << std::endl;
    }

    std::cout << "Benchmark Result" << std::endl;

    double Mean = GetMean(Sample);
    double Variance = GetVariance(Sample);
    double SD = sqrt(Variance);

    std::cout << "Maximum:\t" << Maximum << std::endl;
    std::cout << "Minimum:\t" << Minimum << std::endl;
    std::cout << "Mean:\t" << Mean << std::endl;
    std::cout << "SD:\t" << SD << std::endl;

    return 0;
}

double GetMean (const std::vector<double> &Sample)
{
    double Sum = 0.0f;

    for (const auto &i : Sample)
    {
        Sum += i;
    }

    return Sum / (double)Sample.size();
}

double GetVariance (const std::vector<double> &Sample)
{
    double Mean = GetMean(Sample);
    double Variance = 0.0f;

    for (const auto &i : Sample)
    {
        Variance += pow(i - Mean, 2);
    }

    return Variance / ((double)Sample.size() - 1);
}