//
// Created by Sippawit Thammawiset on 27/1/2024 AD.
//

#ifndef IGWO_H
#define IGWO_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

/**
 * @brief A macro to clamp a value between a minimum and a maximum.
 */
#define CLAMP(X, MIN, MAX)                      std::max(MIN, std::min(MAX, X))

namespace MTH::IGWO
{
    namespace STATE
    {
        /**
         * @brief An enum defining the states of the algorithm.
         */
        enum
        {
            FAILED = 0,
            SUCCESS = 1
        };
    }

    /**
     * @brief Generate a random number within a specified range.
     *
     * @param LowerBound Lower bound of the range.
     * @param UpperBound Upper bound of the range.
     * @return Random number within the specified range.
     */
    double GenerateRandom (double LowerBound = 0.0, double UpperBound = 1.0)
    {
        std::random_device Engine;
        std::uniform_real_distribution<double> RandomDistribution(0.0, 1.0);
        return LowerBound + RandomDistribution(Engine) * (UpperBound - LowerBound);
    }

    /**
     * @brief Generate a random integer between 0 and N.
     *
     * @param N Upper bound of the range.
     * @return Random integer between 0 and N.
     */
    int GenerateRandom (int N)
    {
        std::random_device Engine;
        std::uniform_int_distribution RandomDistribution(0, N);
        return RandomDistribution(Engine);
    }

    /**
     * @brief A struct representing a wolf in the IGWO algorithm.
     */
    template <typename T>
    struct AWolf
    {
        AWolf () : Cost(2e10) {}

        std::vector<T> Position;
        double Cost;
    };

    /**
     * @brief A struct representing leader wolves in the IGWO algorithm.
     */
    template <typename T>
    struct ALeaderWolf
    {
        AWolf<T> Alpha;    // 1st Best
        AWolf<T> Beta;     // 2nd Best
        AWolf<T> Delta;    // 3rd Best
    };

    /**
     * @brief A class representing the Improved Grey Wolf Optimizer (IGWO) algorithm.
     */
    template <typename T>
    class AIGWO
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param LowerBound Lower bound of the search space.
         * @param UpperBound Upper bound of the search space.
         * @param MaximumIteration Maximum number of iterations.
         * @param NPopulation Population size.
         * @param NVariable Number of variables.
         * @param MaximumWeight The maximum weight.
         * @param MinimumWeight The minimum weight.
         * @param Log Flag indicating whether to log information during optimization.
         */
        inline AIGWO <T> (const std::vector<double> &LowerBound, const std::vector<double> &UpperBound,
                          int MaximumIteration, int NPopulation, int NVariable,
                          double MaximumWeight = 2.2f, double MinimumWeight = 0.02f,
                          bool Log = true) :
                          LowerBound_(LowerBound),
                          UpperBound_(UpperBound),
                          MaximumIteration_(MaximumIteration),
                          NPopulation_(NPopulation),
                          NVariable_(NVariable),
                          MaximumWeight_(MaximumWeight),
                          MinimumWeight_(MinimumWeight),
                          Log_(Log)
        {

        }

        /**
         * @brief Destructor.
         */
        inline ~AIGWO <T> () = default;

        /**
         * @brief Set the objective function for optimization.
         *
         * @param UserObjectiveFunction Pointer to the objective function.
         */
        void SetObjectiveFunction (double (*UserObjectiveFunction)(const std::vector<T> &Position))
        {
            this->ObjectiveFunction_ = UserObjectiveFunction;
        }

        /**
         * @brief Run the IGWO algorithm.
         *
         * @return Success or failure flag.
         */
        bool Run ()
        {
            // Check if the objective function is defined
            if (ObjectiveFunction_ == nullptr)
            {
                std::cerr << "Objective function is not defined. Please use SetObjectiveFunction(UserObjectiveFunction) before calling Run()." << std::endl;

                return STATE::FAILED;
            }

            // Check if the size of lower bound or upper bound matches the number of variables
            if (this->LowerBound_.size() != NVariable_ || this->UpperBound_.size() != NVariable_)
            {
                std::cerr << "Size of lowerbound or upperbound does not match with number of variables." << std::endl;

                return STATE::FAILED;
            }

            // Initialize population
            this->Population_ = std::vector<AWolf<T>> (NPopulation_);

            // Initialize wolves with random positions
            for (int PopulationIndex = 0; PopulationIndex < this->NPopulation_; PopulationIndex++)
            {
                auto *CurrentPopulation = &this->Population_[PopulationIndex];

                std::vector<T> Position(this->NVariable_);

                // Generate random positions within bounds for each variable
                for (int VariableIndex = 0; VariableIndex < this->NVariable_; VariableIndex++)
                {
                    T RandomPosition = GenerateRandom(this->LowerBound_[VariableIndex], this->UpperBound_[VariableIndex]);

                    Position[VariableIndex] = RandomPosition;
                }

                CurrentPopulation->Position = Position;

                double Cost = ObjectiveFunction_(CurrentPopulation->Position);
                CurrentPopulation->Cost = Cost;
            }

            // Run optimization process for a maximum number of iterations
            for (int Iteration = 1; Iteration <= this->MaximumIteration_; Iteration++)
            {
                // Update the global best position
                UpdateGlobalBestPosition();

                // Calculate weight
                CalculateWeight(Iteration);

                // Optimize wolves positions
                Optimize();

                this->GlobalBestPosition_ = this->GlobalBest_.Alpha.Position;
                this->GlobalBestCost_ = this->GlobalBest_.Alpha.Cost;

                if (this->Log_)
                {
                    std::cout << "[INFO] Iteration: " << Iteration << " >>>\t"
                              << "Best Cost: " << this->GlobalBestCost_ <<
                              std::endl;
                }
            }

            std::cout << "[INFO] Completed." << std::endl;

            return STATE::SUCCESS;
        }

        /**
         * @brief Get the global best position found by the algorithm.
         *
         * @return Global best position.
         */
        std::vector<T> GetGlobalBestPosition () const
        {
            return this->GlobalBestPosition_;
        }

        /**
         * @brief Get the global best cost found by the algorithm.
         *
         * @return Global best cost.
         */
        double GetGlobalBestCost () const
        {
            return this->GlobalBestCost_;
        }

    private:
        std::vector<T> LowerBound_; /**< Lower bound of the search space. */
        std::vector<T> UpperBound_; /**< Upper bound of the search space. */
        int MaximumIteration_; /**< Maximum number of iterations. */
        int NPopulation_; /**< Population size. */
        int NVariable_; /**< Number of variables. */
        double AlphaWeight_; /**< Weight for alpha wolf */
        double BetaWeight_; /**< Weight for beta wolf */
        double DeltaWeight_; /**< Weight for delta wolf */
        double AlphaGrowthFactor_ = 2.0; /**< Growth factor for alpha wolf weight. */
        double DeltaGrowthFactor_ = 3.0; /**< Growth factor for delta wolf weight. */
        double MaximumWeight_; /**< Maximum weight */
        double MinimumWeight_; /**< Minimum weight */

        double (*ObjectiveFunction_)(const std::vector<T> &Position) = nullptr; /**< Pointer to the objective function. */

        std::vector<AWolf<T>> Population_; /**< Vector containing wolves. */

        ALeaderWolf<T> GlobalBest_; /**< Global best wolves found by the algorithm. */
        std::vector<T> GlobalBestPosition_; /** Global best position found by the algorithm. */
        double GlobalBestCost_ ; /** Global best cost found by the algorithm */

        bool Log_; /**< Flag indicating whether to log information during optimization. */

        /**
         * @brief Update the global best position.
         *
         * This function iterates through the population and updates the global best position
         * based on the cost of each individual in the population.
         */
        void UpdateGlobalBestPosition ()
        {
            // Iterate over each wolf in the population
            for (int PopulationIndex = 0; PopulationIndex < this->NPopulation_; PopulationIndex++)
            {
                auto *CurrentPopulation = &this->Population_[PopulationIndex];

                // Update alpha wolf's position and cost if the current wolf has a lower cost than alpha's cost
                if (CurrentPopulation->Cost < GlobalBest_.Alpha.Cost)
                {
                    GlobalBest_.Alpha.Position = CurrentPopulation->Position;
                    GlobalBest_.Alpha.Cost = CurrentPopulation->Cost;
                }

                // Update beta wolf's position and cost if the current wolf has a lower cost than beta's cost
                if (CurrentPopulation->Cost > GlobalBest_.Alpha.Cost &&
                    CurrentPopulation->Cost < GlobalBest_.Beta.Cost)
                {
                    GlobalBest_.Beta.Position = CurrentPopulation->Position;
                    GlobalBest_.Beta.Cost = CurrentPopulation->Cost;
                }

                // Update delta wolf's position and cost if the current wolf has a lower cost than delta's cost
                if (CurrentPopulation->Cost > GlobalBest_.Alpha.Cost &&
                    CurrentPopulation->Cost > GlobalBest_.Beta.Cost &&
                    CurrentPopulation->Cost < GlobalBest_.Delta.Cost)
                {
                    GlobalBest_.Delta.Position = CurrentPopulation->Position;
                    GlobalBest_.Delta.Cost = CurrentPopulation->Cost;
                }
            }
        }

        /**
         * @brief Calculate the weights based on the current iteration.
         *
         * This function calculates the weights (alpha, beta, and delta)
         * based on the current iteration.
         *
         * @param Iteration The current iteration.
         *
         * @note The weight calculation is extracted from the paper "Improved GWO algorithm
         * for optimal design of truss structures" by A. Kaveh and P. Zakian.
         * The weight equations are derived from Equation (8), (9), and (10) of their paper.
         * Link to the paper: https://link.springer.com/article/10.1007/s00366-017-0567-1
         */
        void CalculateWeight (int Iteration)
        {
            // Calculate alpha weight
            this->AlphaWeight_ = this->MaximumWeight_ *
                                 exp(pow(static_cast<double>(Iteration) / this->MaximumIteration_, this->AlphaGrowthFactor_) *
                                     log(this->MinimumWeight_ / this->MaximumWeight_));

            // Calculate delta weight
            this->DeltaWeight_ = this->MaximumWeight_ *
                                 exp(pow(static_cast<double>(Iteration) / this->MaximumIteration_, this->DeltaGrowthFactor_) *
                                     log(this->MinimumWeight_ / this->MaximumWeight_));

            // Calculate beta weight as the average of alpha and delta weights
            this->BetaWeight_ = (this->AlphaWeight_ + this->DeltaWeight_) * 0.5;
        }

        /**
         * @brief Optimize the positions of wolves in the population.
         *
         * This function optimizes the positions of wolves in the population.
         * It calculates the GWO and DLH positions of each wolf, evaluates the cost of the calculated positions,
         * updates the best position of each wolf based on the comparison of GWO and DLH costs.
         *
         * @note The IGWO algorithm incorporates an additional movement strategy named
         * dimension learning-based hunting (DLH) search strategy.
         * In DLH, each individual wolf is learned by its neighbors to be another candidate
         * for the new position of Xi (t). The information about IGWO algorithm and DLH is based on the paper "An improved grey wolf optimizer for solving engineering problems"
         * by Mohammad H. Nadimi-Shahraki, Shokooh Taghian, and Seyedali Mirjalili.
         * Link to the paper: https://www.sciencedirect.com/science/article/pii/S0957417420307107
         */
        void Optimize ()
        {
            // Iterate over each wolf in the population
            for (int PopulationIndex = 0; PopulationIndex < this->NPopulation_; PopulationIndex++)
            {
                auto *CurrentPopulation = &this->Population_[PopulationIndex];

                double Radius = 0.0;
                std::vector<double> GWOPosition(this->NVariable_);

                // Iterate over each variable of the wolf
                for (int VariableIndex = 0; VariableIndex < this->NVariable_; VariableIndex++)
                {
                    // Generate random values for alpha, beta, and delta
                    double A1 = this->AlphaWeight_ * (2.0 * GenerateRandom(0.0, 1.0) - 1.0);
                    double A2 = this->BetaWeight_ * (2.0 * GenerateRandom(0.0, 1.0) - 1.0);
                    double A3 = this->DeltaWeight_ * (2.0 * GenerateRandom(0.0, 1.0) - 1.0);

                    // Generate random values for C1, C2, and C3
                    double C1 = 2.0 * GenerateRandom(0.0, 1.0);
                    double C2 = 2.0 * GenerateRandom(0.0, 1.0);
                    double C3 = 2.0 * GenerateRandom(0.0, 1.0);

                    // Calculate new position components
                    double X1 = GlobalBest_.Alpha.Position[VariableIndex] - A1 *
                                abs(C1 * GlobalBest_.Alpha.Position[VariableIndex] - CurrentPopulation->Position[VariableIndex]);
                    double X2 = GlobalBest_.Beta.Position[VariableIndex] - A2 *
                                abs(C2 * GlobalBest_.Beta.Position[VariableIndex] - CurrentPopulation->Position[VariableIndex]);
                    double X3 = GlobalBest_.Delta.Position[VariableIndex] - A3 *
                                abs(C3 * GlobalBest_.Delta.Position[VariableIndex] - CurrentPopulation->Position[VariableIndex]);

                    // Calculate new position
                    double X = (X1 + X2 + X3) / 3.0;
                    double NewPosition = X;

                    // Clamp the new position to stay within specified bounds
                    NewPosition = CLAMP(NewPosition, this->LowerBound_[VariableIndex], this->UpperBound_[VariableIndex]);

                    // Equation (10)
                    Radius += std::hypot(CurrentPopulation->Position[VariableIndex] - NewPosition, 0.0);

                    // Store GWO position
                    GWOPosition[VariableIndex] = NewPosition;
                }

                // Get the index of neighborhood and calculate the DLH position
                std::vector<int> Index = GetNeighborHoodIndex(CurrentPopulation, Radius);
                std::vector<double> DLHPosition = CalculateDLHPosition(CurrentPopulation, Index);

                // Calculate costs for GWO and DLH positions
                double GWOCost = ObjectiveFunction_(GWOPosition);
                double DLHCost = ObjectiveFunction_(DLHPosition);

                // Update the wolf's position and cost based on the comparison of GWO and DLH costs
                // Equation (13)
                if (GWOCost < DLHCost)
                {
                    CurrentPopulation->Position = GWOPosition;
                    CurrentPopulation->Cost = GWOCost;
                }
                else
                {
                    CurrentPopulation->Position = DLHPosition;
                    CurrentPopulation->Cost = DLHCost;
                }
            }
        }

        /**
         * @brief Get the indices of wolves within the neighborhood of a given wolf.
         *
         * This function calculates the indices of wolves within the neighborhood of a given wolf
         * based on the provided radius.
         *
         * @param CurrentPopulation Pointer to the current population.
         * @param Radius The radius used to define the neighborhood.
         *
         * @return A vector containing the indices of wolves within the neighborhood.
         */
        std::vector<int> GetNeighborHoodIndex (AWolf<T> *CurrentPopulation, double &Radius)
        {
            std::vector<int> Index;

            // Iterate over each wolf in the population
            for (int PopulationIndex = 0; PopulationIndex < this->NPopulation_; PopulationIndex++)
            {
                auto *NeighborPopulation = &this->Population_[PopulationIndex];

                double Distance = 0.0;

                // Calculate the distance between the current wolf and neighbor wolf in the population
                for (int VariableIndex = 0; VariableIndex < this->NVariable_; VariableIndex++)
                {
                    Distance += std::hypot(CurrentPopulation->Position[VariableIndex] - NeighborPopulation->Position[VariableIndex],
                                           0.0f);
                }

                // Equation (11)
                if (Distance <= Radius)
                {
                    Index.push_back(PopulationIndex);
                }
            }

            return Index;
        }

        /**
         * @brief Calculate the DLH position for a wolf.
         *
         * This function calculates the DLH position for a wolf based on its neighbors' positions.
         *
         * @param CurrentPopulation Pointer to the current population.
         * @param Index Vector containing the indices of wolves within the neighborhood.
         *
         * @return A vector representing the DLH position for the current wolf.
         */
        std::vector<double> CalculateDLHPosition (AWolf<T> *CurrentPopulation, const std::vector<int> &Index)
        {
            std::vector<double> DLHPosition(this->NVariable_);

            // Iterate over each wolf in the population
            for (int VariableIndex = 0; VariableIndex < this->NVariable_; VariableIndex++)
            {
                // Select a random index from the neighborhood
                int NeighborIndex = GenerateRandom((int)Index.size() - 1);

                // Select a random index from the population
                int PopulationIndex = GenerateRandom(this->NPopulation_ - 1);

                // Equation (12)
                double DLH = CurrentPopulation->Position[VariableIndex] +
                             GenerateRandom(0.0f, 1.0f) *
                             (this->Population_[Index[NeighborIndex]].Position[VariableIndex] -
                              this->Population_[PopulationIndex].Position[VariableIndex]);

                // Clamp the DLH position to stay within specified bounds
                DLH = CLAMP(DLH, this->LowerBound_[VariableIndex], this->UpperBound_[VariableIndex]);

                // Store DLH position
                DLHPosition[VariableIndex] = DLH;
            }

            return DLHPosition;
        }
    };
} // MTH

#endif //IGWO_H
