#ifndef IO_H
#define IO_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
//#include <format> //commented out by GK
#include <map>
#include <utility> // std::pair
#include <iostream>

// Local includes
#include "data_structures.h"

// Function prototypes
std::vector<std::pair<std::string, std::vector<double> > > read_csv(std::string filename);
void record_totals(const std::map<std::string, double> &totals, const std::string &filename, const std::string &mode);
void record_sequences_clones(double t, const std::vector<Sequence> &sequences, const std::string &seq_filename, const std::string &clone_filename, const std::string &mode);
void record_defectives(double t, const std::vector<Defective> &defectives, const std::string &filename, const std::string &mode);

#endif // IO_H
