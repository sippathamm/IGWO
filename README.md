# Improved Grey Wolf Optimizer (IGWO) algorithm

## Overview

The IGWO algorithm presents a refined version of the Grey Wolf Optimizer (GWO), a metaheuristic algorithm inspired by the social hierarchy and hunting behavior of grey wolves. 
IGWO builds upon the foundation of GWO, enhancing its capabilities for optimizing complex problems by introducing innovative strategies for balancing exploration and exploitation of the solution space. 
Through the integration of novel mechanisms, IGWO aims to achieve superior performance and convergence rates compared to its predecessor, offering a potent tool for solving a wide range of optimization tasks.

## Features

- Implementation of the IGWO algorithm in C++.
- Easy-to-use interface for optimizing user-defined objective functions.
- Customizable parameters for fine-tuning the optimization process.
- Example code demonstrating the usage of the IGWO algorithm for solving optimization problems.

## Getting Started

Follow these steps to get started with the IGWO algorithm:

1. **Clone the Repository**: Clone this repository to your local machine using the following command:

    ```bash
    git clone https://github.com/sippathamm/IGWO.git
    ```

2. **Build the Project**: Navigate to the root directory and create a build directory. Then, run CMake to configure the project and generate build files:

    ```bash
    cd IGWO
    mkdir build
    cmake -S . -B build
    ```

3. **Compile the Code**: Once the build files are generated, build the IGWO executable by running:

    ```bash
    cmake --build build --target IGWO
    ```

4. **Run the Executable**: Execute the IGWO algorithm by navigating to the build directory and running the generated executable:

    ```bash
    cd build
    ./IGWO
    ```
   
## Usage

Replace the `ObjectiveFunction` implementation with your own objective function logic. This function should take a `std::vector<double>` representing the position of the particle in the search space and return the value of the objective function at that position.

Do not forget to call `IGWO.SetObjectiveFunction(ObjectiveFunction)` before calling `IGWO.Run()`

## Example

This example demonstrates the usage of the IGWO algorithm to find the global minimum with the following parameters:
- Maximum iteration: 1000
- Number of population: 50
- Number of dimensions: 30
- Lowerbound: -100
- Upperbound: 100
- Maximum weight: 2.2
- Minimum weight: 0.02

The objective function used in this example is a Sphere function.

```cpp
#include <iostream>

#include "IGWO.h"

// Define your objective function here
double ObjectiveFunction (const std::vector<double> &Position) 
{
    // This function should return the value of the objective function at the given position
    
    double Sum = 0.0;

    for (const double &i : Position)
    {
        Sum += i * i;
    }

    return Sum;
}

int main () 
{
    // Initialize parameters
    int MaximumIteration = 1000;
    int NPopulation = 50;
    int NVariable = 30;
    std::vector<double> LowerBound = std::vector<double> (NVariable, -100);
    std::vector<double> UpperBound = std::vector<double> (NVariable, 100);
    double MaximumWeight = 2.2;
    double MinimumWeight = 0.02;

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
    }
    else // If the algorithm fails to run
    {
        break; // Do nothing
    }
        
    return 0;
}
```

## Feedback and Bugs Report

If you have any feedback, suggestions, or encounter bugs while using the IGWO algorithm, please feel free to open an issue on the [GitHub repository](https://github.com/sippathamm/IGWO/issues).

## Author

This repository is maintained by Sippawit Thammawiset. You can contact the author at sippawit.t@kkumail.com
