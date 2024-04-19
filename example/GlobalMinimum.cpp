//
// Created by Sippawit Thammawiset on 27/1/2024 AD.
//

#include <iostream>

#include "BenchmarkFunction.h"
#include "IGWO.h"

/**
 * @brief A macro to select pre-implemented benchmark function.
 */
#define BENCHMARK_NAME  Benchmark::SCHWEFEL_S_2_26

double GetMean(const std::vector<double> &Sample);
double GetVariance(const std::vector<double> &Sample);

/**
 * @brief Objective function to be optimized by the algorithm.
 *
 * This function defines the objective function to be optimized by the algorithm.
 * Users should implement their own objective function according to their optimization problem.
 * The function evaluates the objective function value at a given position.
 *
 * @param Position The position vector at which the objective function is to be evaluated.
 * @return The objective function value or cost at the given position.
 */
double ObjectiveFunction (const std::vector<double> &Position)
{
    // Implement your objective function here
    // e.g.
    // for (const auto &i : Position)
    // {
    //      Perform calculation on position
    // }

    return Benchmark::BenchmarkFunction(BENCHMARK_NAME, Position);
}

int main()
{
    std::vector<double> LowerBound, UpperBound;
    int MaximumIteration, NPopulation, NVariable;
    double MaximumWeight = 2.2, MinimumWeight = 0.02;

    // Get benchmark properties
    Benchmark::BenchmarkCondition(BENCHMARK_NAME,
                                  LowerBound, UpperBound,
                                  MaximumIteration, NPopulation, NVariable);

    int NRun = 30; // Number of runs for benchmarking

    // Variables for the results
    double Maximum = -INFINITY;
    double Minimum = INFINITY;
    std::vector<double> Sample;

    // Run the algorithm for multiple runs
    for (int Run = 1; Run <= NRun; ++Run)
    {
        std::cout << "-------- " << "Run " << Run << " --------" << std::endl;

        // Initialize IPSO algorithm
        MTH::IGWO::AIGWO<double> IGWO(LowerBound, UpperBound,
                                      MaximumIteration, NPopulation, NVariable,
                                      MaximumWeight, MinimumWeight,
                                      false);

        // Set objective function for the algorithm
        IGWO.SetObjectiveFunction(ObjectiveFunction);

        if (IGWO.Run()) // If the algorithm runs successfully
        {
            auto GlobalBestPosition = IGWO.GetGlobalBestPosition();

            std::cout << "Global Best Position:\t";
            std::for_each(GlobalBestPosition.begin(), GlobalBestPosition.end(), [](const auto &i) { std::cout << i << "\t"; });
            std::cout << std::endl;

            double GlobalBestCost = IGWO.GetGlobalBestCost();
            std::cout << "Global Best Cost:\t" << GlobalBestCost << std::endl;

            Maximum = std::max(Maximum, GlobalBestCost);
            Minimum = std::min(Minimum, GlobalBestCost);

            Sample.push_back(GlobalBestCost);
        }
        else // If the algorithm fails to run
        {
            break; // Do nothing
        }
    }

    std::cout << "-------- " << "Benchmark Result" << " --------" << std::endl;

    // Variables for the statistics
    double Mean = GetMean(Sample);
    double Variance = GetVariance(Sample);
    double SD = sqrt(Variance);

    std::cout << "Maximum:\t" << Maximum << std::endl;
    std::cout << "Minimum:\t" << Minimum << std::endl;
    std::cout << "Mean:\t" << Mean << std::endl;
    std::cout << "SD:\t" << SD << std::endl;

    return 0;
}

/**
 * @brief Calculate the mean of a sample.
 *
 * This function calculates the mean of a given sample of data.
 *
 * @param Sample The vector containing the sample data.
 * @return The mean of the sample.
 */
double GetMean (const std::vector<double> &Sample)
{
    // Calculate sum of all elements in the sample
    double Sum = std::accumulate(Sample.begin(), Sample.end(), 0.0);

    return Sum / static_cast<double>(Sample.size());
}

/**
 * @brief Calculate the variance of a sample.
 *
 * This function calculates the variance of a given sample of data.
 *
 * @param Sample The vector containing the sample data.
 * @return The variance of the sample.
 */
double GetVariance (const std::vector<double> &Sample)
{
    double Mean = GetMean(Sample);
    double Variance = 0.0;

    // Calculate squared differences from the mean
    for (const auto &i : Sample)
    {
        Variance += (i - Mean) * (i - Mean);
    }

    return Sample.size() < 2 ? 0.0 : Variance / static_cast<double>(Sample.size() - 1);
}