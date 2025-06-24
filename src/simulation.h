#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <format>
#include <map>
#include <utility> // std::pair
#include <iostream>

// Local includes
#include "data_structures.h"
#include "config.h"             // model parameters and file names
#include "io.h"                 // file input/output

// Function prototypes
void clear_totals(std::map<std::string, double> &totals);
void initialize_clones(std::vector<Sequence> &sequences, 
                       std::vector<Defective> &defectives,
                       std::map<std::string, double> &totals,
                       const std::map<std::string, double> &params);
void update_clones(std::vector<Sequence> &sequences,
                   std::vector<Defective> &defectives,
                   std::map<std::string, double> &totals,
                   const std::map<std::string, double> &params);
void evolve_clones(std::vector<Sequence> &sequences,
                   std::vector<Defective> &defectives,
                   double beta, double T, double dt);

#endif // SIMULATION_H
