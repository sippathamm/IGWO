//
// Created by Sippawit Thammawiset on 27/1/2024 AD.
//

/* TODO:    - Add comments and documentation
 */

#ifndef GWO_H
#define GWO_H

#include <iostream>
#include <random>

#define CLAMP(X, MIN, MAX)                      std::max(MIN, std::min(MAX, X))
#define IS_OUT_OF_BOUND(X, MIN, MAX)            X < MIN || X > MAX

namespace Optimizer
{
    enum
    {
        FAILED = 0,
        SUCCESS = 1
    };

    double GenerateRandom (double LowerBound = 0.0f, double UpperBound = 1.0f)
    {
        std::random_device Engine;
        std::uniform_real_distribution<double> RandomDistribution(0.0f, 1.0f);
        return LowerBound + RandomDistribution(Engine) * (UpperBound - LowerBound);
    }
    
    int GenerateRandomIndex (int Index)
    {
        std::random_device Engine;
        std::uniform_int_distribution RandomDistribution(0, Index - 1);
        return RandomDistribution(Engine);
    }

    typedef struct AWolf
    {
        AWolf () : Cost(0.0f) {}

        std::vector<double> Position;
        double Cost;
    } AWolf;

    typedef struct ALeaderWolf
    {
        AWolf Alpha;    // 1st Best
        AWolf Beta;     // 2nd Best
        AWolf Delta;    // 3rd Best
    } ALeaderWolf;

    class AGWO
    {
    public:
        AGWO (const std::vector<double> &LowerBound, const std::vector<double> &UpperBound,
              int MaximumIteration, int NPopulation, int NVariable,
              double Theta, double K,
              double Maximum_a = 2.2f, double Minimum_a = 0.02f,
              double MaximumWeight = 0.9f, double MinimumWeight = 0.4f,
              double VelocityFactor = 0.5f,
              bool Log = true) :
              LowerBound_(LowerBound),
              UpperBound_(UpperBound),
              MaximumIteration_(MaximumIteration),
              NPopulation_(NPopulation),
              NVariable_(NVariable),
              Theta_(Theta),
              K_(K),
              Maximum_a_(Maximum_a),
              Minimum_a_(Minimum_a),
              MaximumWeight_(MaximumWeight),
              MinimumWeight_(MinimumWeight),
              VelocityFactor_(VelocityFactor),
              Log_(Log)
        {

        }

        ~AGWO () = default;

        void SetObjectiveFunction (double (*UserObjectiveFunction)(const std::vector<double> &Position))
        {
            this->ObjectiveFunction_ = UserObjectiveFunction;
        }

        bool Run ()
        {
            if (this->ObjectiveFunction_ == nullptr)
            {
                std::cerr << "Objective function is not defined. Please use SetObjectiveFunction(UserObjectiveFunction) before calling Run()." << std::endl;

                return FAILED;
            }

            if (this->LowerBound_.size() != NVariable_ || this->UpperBound_.size() != NVariable_)
            {
                std::cerr << "Size of lowerbound or upperbound does not match with number of variables." << std::endl;

                return FAILED;
            }

            // Initialize
            this->Population_ = std::vector<AWolf> (NPopulation_);

            this->MaximumVelocity_ = std::vector<double> (this->NVariable_);
            this->MinimumVelocity_ = std::vector<double> (this->NVariable_);

            for (int VariableIndex = 0; VariableIndex < this->NVariable_; VariableIndex++)
            {
                this->MaximumVelocity_[VariableIndex] = this->VelocityFactor_ * (this->UpperBound_[VariableIndex] - this->LowerBound_[VariableIndex]);
                this->MinimumVelocity_[VariableIndex] = -this->MaximumVelocity_[VariableIndex];
            }

            for (int PopulationIndex = 0; PopulationIndex < this->NPopulation_; PopulationIndex++)
            {
                auto *CurrentPopulation = &this->Population_[PopulationIndex];

                std::vector<double> Position(this->NVariable_);

                for (int VariableIndex = 0; VariableIndex < this->NVariable_; VariableIndex++)
                {
                    double RandomPosition = GenerateRandom(this->LowerBound_[VariableIndex], this->UpperBound_[VariableIndex]);

                    Position[VariableIndex] = RandomPosition;
                }

                CurrentPopulation->Position = Position;

                double Cost = ObjectiveFunction_(CurrentPopulation->Position);
                CurrentPopulation->Cost = Cost;

                this->AverageCost_ += Cost;
            }

            this->AverageCost_ /= this->NPopulation_;

            // Optimize
            for (int Iteration = 1; Iteration <= this->MaximumIteration_; Iteration++)
            {
                // Update Alpha, Beta, and Delta wolf
                UpdateGlobalBestPosition();

                // Calculate a_Alpha, a_Beta, a_Delta
                Calculate_a(Iteration);

                Optimize();

                if (this->Log_)
                {
                    std::cout << "[INFO] Iteration: " << Iteration << " >>> " << "Best Cost: "
                              << this->GlobalBestPosition_.Alpha.Cost << std::endl;
                }

                this->NextAverageCost_ /= this->NPopulation_;
                this->AverageCost_ = this->NextAverageCost_;
                this->NextAverageCost_ = 0.0f;
            }

            std::cout << "[INFO] Completed." << std::endl;

            return SUCCESS;
        }

        std::vector<double> GetGlobalBestPosition () const
        {
            return this->GlobalBestPosition_.Alpha.Position;
        }

        double GetGlobalBestCost () const
        {
            return this->GlobalBestPosition_.Alpha.Cost;
        }

    private:
        std::vector<double> LowerBound_, UpperBound_;
        int MaximumIteration_, NPopulation_, NVariable_;
        double Theta_, K_;
        double Weight_{}, H_{};
        double a_Alpha_{}, a_Beta_{}, a_Delta_{};
        double AlphaGrowthFactor_ = 2.0f, DeltaGrowthFactor_ = 3.0f;
        double Maximum_a_, Minimum_a_;
        double MaximumWeight_, MinimumWeight_;
        double VelocityFactor_;

        double (*ObjectiveFunction_)(const std::vector<double> &Position) = nullptr;

        std::vector<AWolf> Population_;
        std::vector<double> MaximumVelocity_, MinimumVelocity_;

        ALeaderWolf GlobalBestPosition_;
        double AverageCost_ = 0.0f;
        double NextAverageCost_ = 0.0f;

        bool Log_;

        void UpdateGlobalBestPosition ()
        {
            for (int PopulationIndex = 0; PopulationIndex < this->NPopulation_; PopulationIndex++)
            {
                auto *CurrentPopulation = &this->Population_[PopulationIndex];

                if (CurrentPopulation->Cost < GlobalBestPosition_.Alpha.Cost)
                {
                    GlobalBestPosition_.Alpha.Position = CurrentPopulation->Position;
                    GlobalBestPosition_.Alpha.Cost = CurrentPopulation->Cost;
                }
                if (CurrentPopulation->Cost > GlobalBestPosition_.Alpha.Cost &&
                    CurrentPopulation->Cost < GlobalBestPosition_.Beta.Cost)
                {
                    GlobalBestPosition_.Beta.Position = CurrentPopulation->Position;
                    GlobalBestPosition_.Beta.Cost = CurrentPopulation->Cost;
                }
                if (CurrentPopulation->Cost > GlobalBestPosition_.Alpha.Cost &&
                    CurrentPopulation->Cost > GlobalBestPosition_.Beta.Cost &&
                    CurrentPopulation->Cost < GlobalBestPosition_.Delta.Cost)
                {
                    GlobalBestPosition_.Delta.Position = CurrentPopulation->Position;
                    GlobalBestPosition_.Delta.Cost = CurrentPopulation->Cost;
                }
            }
        }

        void Calculate_a (int Iteration)
        {
            this->a_Alpha_ = this->Maximum_a_ *
                             exp(pow(static_cast<double>(Iteration) / this->MaximumIteration_, this->AlphaGrowthFactor_) *
                             log(this->Minimum_a_ / this->Maximum_a_));
            this->a_Delta_ = this->Maximum_a_ *
                             exp(pow(static_cast<double>(Iteration) / this->MaximumIteration_, this->DeltaGrowthFactor_) *
                             log(this->Minimum_a_ / this->Maximum_a_));
            this->a_Beta_ = (this->a_Alpha_ + this->a_Delta_) * 0.5f;
        }
        
        void CalculateWeight (AWolf *CurrentPopulation)
        {
            this->H_ = this->K_ * ((0.0f - CurrentPopulation->Cost) / this->AverageCost_);
            //                      ^ Optimal cost
            this->Weight_ = ((this->MaximumWeight_ + this->MinimumWeight_) * 0.5f) +
                            (this->MaximumWeight_ - this->MinimumWeight_) * (this->H_ / (std::hypot(1.0f, this->H_))) * tan(this->Theta_);
        }

        void Optimize ()
        {
            for (int PopulationIndex = 0; PopulationIndex < this->NPopulation_; PopulationIndex++)
            {
                auto *CurrentPopulation = &this->Population_[PopulationIndex];

                CalculateWeight(CurrentPopulation);

                double Radius = 0.0f;
                std::vector<double> GWOPosition(this->NVariable_);

                for (int VariableIndex = 0; VariableIndex < this->NVariable_; VariableIndex++)
                {
                    double A1 = this->a_Alpha_ * (2.0f * GenerateRandom(0.0f, 1.0f) - 1);
                    double A2 = this->a_Beta_ * (2.0f * GenerateRandom(0.0f, 1.0f) - 1);
                    double A3 = this->a_Delta_ * (2.0f * GenerateRandom(0.0f, 1.0f) - 1);

                    double C1 = 2.0f * GenerateRandom(0.0f, 1.0f);
                    double C2 = 2.0f * GenerateRandom(0.0f, 1.0f);
                    double C3 = 2.0f * GenerateRandom(0.0f, 1.0f);

                    double X1 = GlobalBestPosition_.Alpha.Position[VariableIndex] - A1 *
                                abs(C1 * GlobalBestPosition_.Alpha.Position[VariableIndex] - CurrentPopulation->Position[VariableIndex]);
                    double X2 = GlobalBestPosition_.Beta.Position[VariableIndex] - A2 *
                                abs(C2 * GlobalBestPosition_.Beta.Position[VariableIndex] - CurrentPopulation->Position[VariableIndex]);
                    double X3 = GlobalBestPosition_.Delta.Position[VariableIndex] - A3 *
                                abs(C3 * GlobalBestPosition_.Delta.Position[VariableIndex] - CurrentPopulation->Position[VariableIndex]);

                    double X = (X1 + X2 + X3) / 3.0f;

                    double R1 = GenerateRandom(0.0f, 1.0f);
                    double R2 = GenerateRandom(0.0f, 1.0f);
                    double R3 = GenerateRandom(0.0f, 1.0f);

                    double NewVelocity = this->Weight_ * ((C1 * R1 * (X1 - X)) + (C2 * R2 * (X2 - X)) + (C3 * R3 * (X3 - X)));

                    NewVelocity = CLAMP(NewVelocity, this->MinimumVelocity_[VariableIndex], this->MaximumVelocity_[VariableIndex]);

                    double TemporaryNewPosition = X + NewVelocity;

                    if (IS_OUT_OF_BOUND(TemporaryNewPosition, this->LowerBound_[VariableIndex], this->UpperBound_[VariableIndex]))
                    {
                        double VelocityConfinement = -GenerateRandom(0.0f, 1.0f) * NewVelocity;

                        NewVelocity = VelocityConfinement;
                    }

                    double NewPosition = X + NewVelocity;

                    NewPosition = CLAMP(NewPosition, this->LowerBound_[VariableIndex], this->UpperBound_[VariableIndex]);

                    // R = (X_i - X_new)^2
                    Radius += std::hypot(CurrentPopulation->Position[VariableIndex] - NewPosition,
                                         0.0f);

                    GWOPosition[VariableIndex] = NewPosition;
                }

                // Stored index which neighbor distance <= radius
                std::vector<int> Index = GetNeighborHoodIndex(CurrentPopulation, Radius);
                std::vector<double> DLHPosition = CalculateDLHPosition(CurrentPopulation, Index);

                double GWOCost = ObjectiveFunction_(GWOPosition);
                double DLHCost = ObjectiveFunction_(DLHPosition);

                if (GWOCost < DLHCost)
                {
                    CurrentPopulation->Position = GWOPosition;
                    CurrentPopulation->Cost = GWOCost;

                    this->NextAverageCost_ += GWOCost;
                }
                else
                {
                    CurrentPopulation->Position = DLHPosition;
                    CurrentPopulation->Cost = DLHCost;

                    this->NextAverageCost_ += DLHCost;
                }
            }
        }

        std::vector<int> GetNeighborHoodIndex (AWolf *CurrentPopulation, double &Radius)
        {
            std::vector<int> Index;

            for (int PopulationIndex = 0; PopulationIndex < this->NPopulation_; PopulationIndex++)
            {
                auto *AnotherPopulation = &this->Population_[PopulationIndex];

                double Distance = 0.0;

                for (int VariableIndex = 0; VariableIndex < this->NVariable_; VariableIndex++)
                {
                    // D = (X_i - X_j)^2
                    Distance += std::hypot(CurrentPopulation->Position[VariableIndex] - AnotherPopulation->Position[VariableIndex],
                                           0.0f);
                }

                if (Distance <= Radius)
                {
                    Index.push_back(PopulationIndex);
                }
            }
            
            return Index;
        }

        std::vector<double> CalculateDLHPosition (AWolf *CurrentPopulation, const std::vector<int> &Index)
        {
            std::vector<double> DLHPosition(this->NVariable_);

            for (int VariableIndex = 0; VariableIndex < this->NVariable_; VariableIndex++)
            {
                int NeighborIndex = GenerateRandomIndex((int)Index.size());
                int PopulationIndex = GenerateRandomIndex(this->NPopulation_);

                double DLH = CurrentPopulation->Position[VariableIndex] +
                             GenerateRandom(0.0f, 1.0f) *
                             (this->Population_[Index[NeighborIndex]].Position[VariableIndex] -
                             this->Population_[PopulationIndex].Position[VariableIndex]);

                DLH = CLAMP(DLH, this->LowerBound_[VariableIndex], this->UpperBound_[VariableIndex]);

                DLHPosition[VariableIndex] = DLH;
            }

            return DLHPosition;
        }
    };
} // Optimizer

#endif //GWO_H
