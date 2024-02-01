//
// Created by Sippawit Thammawiset on 27/1/2024 AD.
//

#ifndef GWO_H
#define GWO_H

#include <iostream>
#include <queue>
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

    typedef struct AWolf
    {
        AWolf () : FitnessValue(INFINITY) {}

        std::vector<double> Position;
        double FitnessValue;
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
        AGWO (const std::vector<double>& LowerBound, const std::vector<double>& UpperBound,
              int MaxIteration, int NPopulation, int NVariable,
              double Theta, double K,
              double VelocityFactor = 0.5f,
              bool Log = true) :
              LowerBound_(LowerBound),
              UpperBound_(UpperBound),
              MaximumIteration_(MaxIteration),
              NPopulation_(NPopulation),
              NVariable_(NVariable),
              Theta_(Theta),
              K_(K),
              VelocityFactor_(VelocityFactor),
              Log_(Log)
        {

        }

        ~AGWO () = default;

        void SetFitnessFunction (double (*UserFitnessFunction)(const std::vector<double> &Position))
        {
            this->FitnessFunction_ = UserFitnessFunction;
        }

        bool Run ()
        {
            if (this->FitnessFunction_ == nullptr)
            {
                std::cerr << "Fitness function is not defined. Please use SetFitnessFunction(UserFitnessFunction) before calling Run()." << std::endl;

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

                double FitnessValue = FitnessFunction_(CurrentPopulation->Position);
                CurrentPopulation->FitnessValue = FitnessValue;

                this->AverageFitnessValue_ += FitnessValue;
            }

            this->AverageFitnessValue_ /= this->NPopulation_;

            // Optimize
            for (int Iteration = 1; Iteration <= this->MaximumIteration_; Iteration++)
            {
                for (int PopulationIndex = 0; PopulationIndex < this->NPopulation_; PopulationIndex++)
                {
                    auto *CurrentPopulation = &this->Population_[PopulationIndex];

                    if (CurrentPopulation->FitnessValue < GlobalBestPosition_.Alpha.FitnessValue)
                    {
                        GlobalBestPosition_.Alpha.Position = CurrentPopulation->Position;
                        GlobalBestPosition_.Alpha.FitnessValue = CurrentPopulation->FitnessValue;
                    }
                    if (CurrentPopulation->FitnessValue > GlobalBestPosition_.Alpha.FitnessValue &&
                        CurrentPopulation->FitnessValue < GlobalBestPosition_.Beta.FitnessValue)
                    {
                        GlobalBestPosition_.Beta.Position = CurrentPopulation->Position;
                        GlobalBestPosition_.Beta.FitnessValue = CurrentPopulation->FitnessValue;
                    }
                    if (CurrentPopulation->FitnessValue > GlobalBestPosition_.Alpha.FitnessValue &&
                        CurrentPopulation->FitnessValue > GlobalBestPosition_.Beta.FitnessValue &&
                        CurrentPopulation->FitnessValue < GlobalBestPosition_.Delta.FitnessValue)
                    {
                        GlobalBestPosition_.Delta.Position = CurrentPopulation->Position;
                        GlobalBestPosition_.Delta.FitnessValue = CurrentPopulation->FitnessValue;
                    }
                }

                double a = 2.0f * (1.0f - (static_cast<double>(Iteration)) / this->MaximumIteration_);

                for (int PopulationIndex = 0; PopulationIndex < this->NPopulation_; PopulationIndex++)
                {
                    auto *CurrentPopulation = &this->Population_[PopulationIndex];

                    this->H_ = this->K_ * ((0.0f - CurrentPopulation->FitnessValue) / this->AverageFitnessValue_);
                    //                      ^ Optimal Fitness Value
                    this->Weight_ = ((this->MaximumWight_ + this->MinimumWeight_) / 2.0f) +
                                    (this->MaximumWight_ - this->MinimumWeight_) * (this->H_ / (std::hypot(1.0f, this->H_))) * tan(this->Theta_);

                    std::vector<double> UpdatedPosition(this->NVariable_);

                    for (int VariableIndex = 0; VariableIndex < this->NVariable_; VariableIndex++)
                    {
                        double X1, X2, X3;

                        double A1 = 2.0f * a * GenerateRandom(0.0f, 1.0f) - a;
                        double C1 = 2.0f * GenerateRandom(0.0f, 1.0f);
                        double D1 = abs(C1 * GlobalBestPosition_.Alpha.Position[VariableIndex] - CurrentPopulation->Position[VariableIndex]);
                        X1 = GlobalBestPosition_.Alpha.Position[VariableIndex] - A1 * D1;

                        double A2 = 2.0f * a * GenerateRandom(0.0f, 1.0f) - a;
                        double C2 = 2.0f * GenerateRandom(0.0f, 1.0f);
                        double D2 = abs(C2 * GlobalBestPosition_.Beta.Position[VariableIndex] - CurrentPopulation->Position[VariableIndex]);
                        X2 = GlobalBestPosition_.Beta.Position[VariableIndex] - A2 * D2;

                        double A3 = 2.0f * a * GenerateRandom(0.0f, 1.0f) - a;
                        double C3 = 2.0f * GenerateRandom(0.0f, 1.0f);
                        double D3 = abs(C3 * GlobalBestPosition_.Delta.Position[VariableIndex] - CurrentPopulation->Position[VariableIndex]);
                        X3 = GlobalBestPosition_.Delta.Position[VariableIndex] - A3 * D3;

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

                        UpdatedPosition[VariableIndex] = NewPosition;
                    }

                    double FitnessValue = FitnessFunction_(UpdatedPosition);
                    this->NextAverageFitnessValue_ += FitnessValue;

                    if (FitnessValue < CurrentPopulation->FitnessValue)
                    {
                        CurrentPopulation->Position = UpdatedPosition;
                        CurrentPopulation->FitnessValue = FitnessValue;
                    }
                }

                if (this->Log_)
                {
                    std::cout << "[INFO] Iteration: " << Iteration << " >>> " << "Best fitness value: " << this->GlobalBestPosition_.Alpha.FitnessValue << std::endl;
                }

                this->NextAverageFitnessValue_ /= this->NPopulation_;
                this->AverageFitnessValue_ = this->NextAverageFitnessValue_;
                this->NextAverageFitnessValue_ = 0.0f;
            }

            std::cout << "[INFO] Completed." << std::endl;

            return SUCCESS;
        }

        std::vector<double> GetGlobalBestPosition () const
        {
            return this->GlobalBestPosition_.Alpha.Position;
        }

        double GetGlobalBestFitnessValue () const
        {
            return this->GlobalBestPosition_.Alpha.FitnessValue;
        }

    private:
        std::vector<double> LowerBound_, UpperBound_;
        int MaximumIteration_, NPopulation_, NVariable_;
        double Weight_{}, H_{}, Theta_, K_;
        double MaximumWight_ = 0.9f, MinimumWeight_ = 0.4f;
        double VelocityFactor_;

        double (*FitnessFunction_)(const std::vector<double> &Position) = nullptr;

        std::vector<AWolf> Population_;
        std::vector<double> MaximumVelocity_, MinimumVelocity_;

        ALeaderWolf GlobalBestPosition_;
        double AverageFitnessValue_ = 0.0f;
        double NextAverageFitnessValue_ = 0.0f;

        bool Log_;
    };
} // Optimizer

#endif //GWO_H
