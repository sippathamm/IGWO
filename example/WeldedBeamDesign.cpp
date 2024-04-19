//
// Created by Sippawit Thammawiset on 25/1/2024 AD.
//

#include <iostream>

#include "BenchmarkFunction.h"
#include "IGWO.h"

double GetMean(const std::vector<double> &Sample);
double GetVariance(const std::vector<double> &Sample);

const double P = 6000.0;
const double E = 30e6;
const double G = 12e6;
const double L = 14.0;

const double PenaltyScalingFactor = 1000.0;
const double TAU_MAX = 13600.0;
const double SIGMA_MAX = 30000.0;
const double DELTA_MAX = 0.25;

double Tau (const std::vector<double> &Position)
{
    double X1 = Position[0];
    double X2 = Position[1];
    double X3 = Position[2];
    double X13Half = (X1 + X3) * 0.5;

    double M = P * (L + X2 * 0.5);
//    double R = sqrt((pow(X2, 2)) / 4 + pow((X1 + X3) / 2, 2));
    double R = sqrt(X2 * X2 * 0.25 + X13Half * X13Half);
    double J = 2 * (sqrt(2.0) * X1 * X2 * ((X2 * X2) / 12.0 + X13Half * X13Half));

    double Tau1 = P / (sqrt(2.0) * X1 * X2);
    double Tau2 = M * R / J;

    return sqrt(Tau1 * Tau1 + Tau1 * Tau2 * X2 / R + Tau2 * Tau2);
}

double Sigma (const std::vector<double> &Position)
{
    double X3 = Position[2];
    double X4 = Position[3];

    return (6 * P * L) / (X4 * X3 * X3);
}

double Delta (const std::vector<double> &Position)
{
    double X3 = Position[2];
    double X4 = Position[3];

    return (4 * P * L * L * L) / (E * X4 * X3 * X3 * X3);
}

double PC (const std::vector<double> &Position)
{
    double X3 = Position[2];
    double X4 = Position[3];

//    double Term1 = 4.013 * E * sqrt((pow(X3, 2) * pow(X4, 6)) / 36);
    double Term1 = 4.013 * E * X3 * X4 * X4 * X4 / 6;
    double Term2 = (1.0 - X3 * sqrt(E / (4 * G)) / (2 * L)) / (L * L);

    return Term1 * Term2;
}

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
    double X1 = Position[0];
    double X2 = Position[1];
    double X3 = Position[2];
    double X4 = Position[3];

    double Cost = 1.10471 * X1 * X1 * X2 + 0.04811 * X3 * X4 * (14.0 + X2);
    double G1 = Tau(Position) - TAU_MAX;
    double G2 = Sigma(Position) - SIGMA_MAX;
    double G3 = X1 - X4;
    double G4 = 0.10471 * X1 * X1 + 0.04811 * X3 * X4 * (14.0 + X2) - 5.0;
    double G5 = 0.125 - X1;
    double G6 = Delta(Position) - DELTA_MAX;
    double G7 = P - PC(Position);
    double Penalty = (std::max(0.0, G1) * std::max(0.0, G1) +
                      std::max(0.0, G2) * std::max(0.0, G2) +
                      std::max(0.0, G3) * std::max(0.0, G3) +
                      std::max(0.0, G4) * std::max(0.0, G4) +
                      std::max(0.0, G5) * std::max(0.0, G5) +
                      std::max(0.0, G6) * std::max(0.0, G6) +
                      std::max(0.0, G7) * std::max(0.0, G7));

    return Cost + PenaltyScalingFactor * Penalty;
}

int main ()
{
    // Initialize parameters
    int MaximumIteration = 1000;
    int NPopulation = 50;
    int NVariable = 4;
    std::vector<double> LowerBound = std::vector<double> (NVariable) = {0.1, 0.1, 0.1, 0.1};
    std::vector<double> UpperBound = std::vector<double> (NVariable) = {2.0, 10.0, 10.0, 2.0};
    double MaximumWeight = 2.2, MinimumWeight = 0.02;

    int NRun = 30; // Number of runs for benchmarking

    // Variables for the results
    double Maximum = -INFINITY;
    double Minimum = INFINITY;
    std::vector<double> Sample;

    // Run the algorithm for multiple runs
    for (int Run = 1; Run <= NRun; ++Run)
    {
        std::cout << "-------- " << "Run " << Run << " --------" << std::endl;

        // Initialize IGWO algorithm
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
