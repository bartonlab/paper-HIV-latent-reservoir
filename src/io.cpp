#include "io.h"


// Read a CSV file into a vector of <string, vector<double>> pairs
// where each pair represents <column name, column values>
std::vector<std::pair<std::string, std::vector<double> > > read_csv(std::string filename){

    // Create a vector of <string, double vector> pairs to store the result
    std::vector<std::pair<std::string, std::vector<double> > > result;

    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");
    
    // Helper vars
    std::string line, colname;
    double val;
    
    // Read the column names
    if(myFile.good()) {
        // Extract the first line in the file
        std::getline(myFile, line);
        // Create a stringstream from line
        std::stringstream ss(line);
        // Extract each column name
        while(std::getline(ss, colname, ',')){
            // Initialize and add <colname, double vector> pairs to result
            result.push_back(std::make_pair(colname,std::vector<double> () ));
        }
    }

    // Read data, line by line
    while(std::getline(myFile, line)) {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        // Keep track of the current column index
        int colIdx = 0;
        // Extract each integer
        while(ss >> val){
            // Add the current integer to the 'colIdx' column's values vector
            result.at(colIdx).second.push_back(val);
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            // Increment the column index
            colIdx++;
        }
    }

    // Close file
    myFile.close();
    return result;

}

// Record totals to a CSV file
void record_totals(const std::map<std::string, double> &totals, const std::string &filename, const std::string &mode) {

    // List of variables to be recorded
    std::vector<std::string> variables = {
        "t", "T", "nLclones", "nMclones", "nDclones", "nSequences",
        "Ltotal", "Atotal", "Vtotal", "LXRtotal", "Lmut", "Amut", "Vmut", "Dtotal",
        "nL", "nmL", "nmA", "nD", "nEvents", "nReact", "Lstim", "Lprof", "Ldeath", "Lreac", "bin1", "bin2", "bin3", "bin4", "bin5"
    };

    // Check if the mode is valid
    if (mode != "a" && mode != "w") {
        throw std::invalid_argument("Invalid mode. Use 'a' for append or 'w' for write.");
    }

    // If mode is "w", create a new file and write the header
    if (mode == "w") {
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            throw std::runtime_error("Could not open file for writing");
        }

        // Write the header
        for (const auto &var : variables) {
            outFile << var << ",";
        }
        outFile.seekp(-1, std::ios_base::cur); // Remove the last comma
        outFile << "\n";
        outFile.close();
    }

    // If mode is "a" or after writing header, append to the existing file
    std::ofstream outFile(filename, std::ios_base::app);
    if (!outFile.is_open()) {
        throw std::runtime_error("Could not open file for appending");
    }

    // Write the values
    for (const auto &var : variables) {
        auto it = totals.find(var);
        if (it != totals.end()) {
            outFile << it->second << ",";
        } else {
            outFile << "0,";
        }
    }
    outFile.seekp(-1, std::ios_base::cur); // Remove the last comma
    outFile << "\n";
    outFile.close();

}

// Record sequences and clones to a CSV file
void record_sequences_clones(double t, const std::vector<Sequence> &sequences, const std::string &seq_filename, const std::string &clone_filename, const std::string &mode) {

    // Check if the mode is valid
    if (mode != "a" && mode != "w") {
        throw std::invalid_argument("Invalid mode. Use 'a' for append or 'w' for write.");
    }

    // If mode is "w", create a new file and write the header
    if (mode == "w") {
        // Create the sequence file
        std::ofstream seqFile(seq_filename);
        if (!seqFile.is_open()) {
            throw std::runtime_error("Could not open file for writing");
        }
        // Write the header
        seqFile << "t,ID_created,ID_sequence,L,A,V,LXR,Nmut,Nclones,nLs,nmLs,nmAs,nDs,nEvents,nReacts\n";
        seqFile.close();

        // Create the clone file
        std::ofstream cloneFile(clone_filename);
        if (!cloneFile.is_open()) {
            throw std::runtime_error("Could not open file for writing");
        }
        // Write the header
        // cloneFile << "t,ID_created,ID_sequence,ID_clone,r,L,Ag,reactivated\n"; //"reactivated" added by GK
        cloneFile << "t,ID_created,ID_sequence,ID_clone,r,L,Ag,Nmut\n"; //added by GK, above commented out by GK
        cloneFile.close();
    }

    // If mode is "a" or after writing header, append to the existing file
    std::ofstream seqFile(seq_filename, std::ios_base::app);
    if (!seqFile.is_open()) {
        throw std::runtime_error("Could not open file for appending");
    }
    std::ofstream cloneFile(clone_filename, std::ios_base::app);
    if (!cloneFile.is_open()) {
        throw std::runtime_error("Could not open file for appending");
    }

    // Iterate through sequences
    for (const auto &seq : sequences) {

        seqFile << t << "," << seq.t_init << "," << seq.id_sequence << "," 
                << seq.L_tot << "," << seq.A << "," << seq.V << "," 
                << seq.LXR << "," << seq.Nmut << "," << seq.clones.size() << "," 
                << seq.n_L << "," << seq.n_Lm << "," << seq.n_Am << "," 
                << seq.n_D << "," << seq.n_infect << "," << seq.n_react << "\n";

        // Iterate through clones
        // for (auto &clone : seq.clones) {
        //     cloneFile << t << "," << clone.t_created << "," << seq.id_sequence << "," 
        //               << clone.id << "," << clone.r << "," << clone.L << "," 
        //               << clone.Ag << "," << clone.reactivated << "\n";
        // }
        //above was original and commented out by GK ("clone.reactivated" added by GK)
        //below written by GK
        for (auto &clone : seq.clones) {
            cloneFile << t << "," << clone.t_created << "," << seq.id_sequence << "," 
                      << clone.id << "," << clone.r << "," << clone.L << "," 
                      << clone.Ag << "," << seq.Nmut << "\n";
        }

    }

    // Close files
    seqFile.close();
    cloneFile.close();

}

// Record defectives to a CSV file
void record_defectives(double t, const std::vector<Defective> &defectives, const std::string &filename, const std::string &mode) {

    // Check if the mode is valid
    if (mode != "a" && mode != "w") {
        throw std::invalid_argument("Invalid mode. Use 'a' for append or 'w' for write.");
    }

    // If mode is "w", create a new file and write the header
    if (mode == "w") {
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            throw std::runtime_error("Could not open file for writing");
        }
        // Write the header
        outFile << "t,ID_created,ID_sequence,D,Ag\n";
        outFile.close();
    }

    // If mode is "a" or after writing header, append to the existing file
    std::ofstream outFile(filename, std::ios_base::app);
    if (!outFile.is_open()) {
        throw std::runtime_error("Could not open file for appending");
    }

    // Iterate through defectives
    for (const auto &def : defectives) {
        outFile << t << "," << def.t_created << "," << def.id_sequence << "," 
                << def.D << "," << def.Ag << "\n";
    }

    // Close file
    outFile.close();

}