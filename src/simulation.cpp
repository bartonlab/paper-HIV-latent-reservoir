#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <vector>
#include <math.h>
#include <iostream>
#include <map>
#include <string>
#include <sstream> // std::stringstream
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <chrono>
//#include <omp.h>
#include <thread>
#include <cstdlib> //added by GK

// Local includes
#include "data_structures.h"    // data structures
#include "config.h"             // model parameters and file names
#include "io.h"                 // file input/output

// GLOBAL VARIABLES
double TIME = 0;
int SEQUENCE_ID = 0; 


// Clear temporary total values
void clear_totals(std::map<std::string, double> &totals) {
    totals["nLclones"] = 0;     // total number of latent clones
    totals["nMclones"] = 0;     // total number of latent clones with mutations AFTER ART
    totals["nDclones"] = 0;     // total number of defective clones
    totals["nSequences"] = 0;   // total number of sequences
    totals["Ltotal"] = 0;       // total number of latent cells
    totals["Atotal"] = 0;       // total number of active cells
    totals["Vtotal"] = 0;       // total number of virions
    totals["LXRtotal"] = 0;     // accumulated sum of latent times reactivation
    totals["Lmut"] = 0;         // number of latent cells with mutations AFTER ART
    totals["Amut"] = 0;         // number of active cells with mutations AFTER ART
    totals["Vmut"] = 0;         // number of virions with mutations AFTER ART
    totals["Dtotal"] = 0;       // total number of defectives
    totals["nL"] = 0;           // new latent clones being produced without mutation
    totals["nmL"] = 0;          // new latent clones being produced with mutation
    totals["nmA"] = 0;          // new active clones being produced with mutation
    totals["nD"] = 0;           // new defective clones being produced
    totals["nEvents"] = 0;      // total number of infection events occurring
    totals["nReact"] = 0;       // number of latent reactivations during that time step
    // totals["Lrep"] = 0;         // number of latent cells replicated during timestep
    totals["Lstim"] = 0;        // number of latent cells produced through antigenic stimulation
    totals["Lprof"] = 0;        // number of latent cells produced through homeostatic proliferation
    totals["Ldeath"] = 0;       // number of latent cells lost through homeostatic death
    totals["Lreac"] = 0;        // number of latent cells lost through homeostatic death
    
    //comment when not doing ATI:
    totals["bin1"] = 0;
    totals["bin2"] = 0;
    totals["bin3"] = 0;
    totals["bin4"] = 0;
    totals["bin5"] = 0;
    
    
}

// Initialize clones
void initialize_clones(std::vector<Sequence> &sequences, 
                       std::vector<Defective> &defectives,
                       std::map<std::string, double> &totals,
                       const std::map<std::string, double> &params) {

    // Future: read in initial conditions from input files
    
    // Start simulation at time zero with 1e4 virions
    std::size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    sequences.push_back(Sequence(seed, params, TIME, SEQUENCE_ID++));
    sequences.back().V = 1e4; // Initial virion count

    // Initialize totals
    clear_totals(totals);
    totals["t"] = TIME;
    totals["T"] = params.at("initial_T");
    totals["nSequences"] = sequences.size();
    totals["Vtotal"] = 1e4;

}

// Evolve all clones
void evolve_clones(std::vector<Sequence> &sequences, 
                   std::vector<Defective> &defectives, 
                   double beta, double T, double dt) {

    // Evolve each sequence
    for (auto &seq : sequences) {
        seq.evolve(beta, T, dt, TIME);
    }

    // Evolve each defective clone
    for (auto &def : defectives) {
        def.evolve(dt);
    }

}

// Remove extinct clones and generate new ones
void update_clones(std::vector<Sequence> &sequences, 
                   std::vector<Defective> &defectives,
                   std::map<std::string, double> &totals,
                   const std::map<std::string, double> &params) {

    // Initialize temporary totals and new sequences/defectives
    clear_totals(totals);
    std::map<int, int> new_latent;
    std::map<int, int> new_active;
    int new_defective=0;

    // Remove extinct sequences
    sequences.erase(std::remove_if(sequences.begin(), sequences.end(), [&totals, &new_latent, &new_active, &new_defective](const Sequence &seq) {
        return seq.update_extinct(totals, new_latent, new_active, new_defective);
    }), sequences.end());

    // Remove extinct defectives
    defectives.erase(std::remove_if(defectives.begin(), defectives.end(), [&totals](const Defective &def) {
        return def.update_extinct(totals);
    }), defectives.end());

    // Create new mutant sequences
    for (const auto &pair : new_latent) {
        size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        int Nmut = pair.first;
        int n_new_latent = pair.second;
        int n_new_active = 0;
        if (new_active.contains(Nmut)) n_new_active += new_active[Nmut];

        if (n_new_latent>0 || n_new_active>0) {
            //check if the sequences vector has an Nmut of that value, and just add it to that
            int check = 0;
            for (Sequence& seq: sequences){
                if (seq.Nmut == Nmut){
                    seq.add_new_latent(n_new_latent, TIME);
                    seq.A += n_new_active;
                    check = 1;
                    break;
                }
            }
            //if the sequences vector doesn't have that specific number of mutations, create a new one
            if (check==0){
                sequences.push_back(Sequence(seed, params, TIME, SEQUENCE_ID++));
                sequences.back().add_new_latent(n_new_latent, TIME);
                sequences.back().A = n_new_active;
                sequences.back().Nmut = Nmut;
                sequences.back().allow_mutations(); // Allow mutations
            }
        }

    }

    for (const auto &pair : new_active) {
        int Nmut = pair.first;
        int n_new_active = pair.second;

        if (!new_latent.contains(Nmut) && n_new_active>0) {
            //check if the sequences vector has an Nmut of that value, and just add it to that
            int check = 0;
            for (Sequence& seq: sequences){
                if (seq.Nmut == Nmut){
                    seq.A += n_new_active;
                    check = 1;
                    break;
                }
            }
            //if the sequences vector doesn't have that specific number of mutations, create a new one
            if (check==0){
                size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
                sequences.push_back(Sequence(seed, params, TIME, SEQUENCE_ID++));
                sequences.back().A = n_new_active;
                sequences.back().Nmut = Nmut; //commented out by GK
                sequences.back().allow_mutations(); // Allow mutations
            }
        }

    }

    // Create new defectives
    for (int i=0; i<new_defective; i++) {
        size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        defectives.push_back(Defective(seed, params, TIME, SEQUENCE_ID++));
    }

}


// MAIN PROGRAM
int main(int argc, char *argv[]) {

    const char* taskIdEnv = std::getenv("SLURM_ARRAY_TASK_ID");
    int taskID = std::stoi(taskIdEnv);

    /* PARAMETERS */

    // Read in parameters
    auto params = getParameters();  // Load all parameters
    auto ioFiles = getFileIO(taskID);     // Load file names

    //uncomment this if not running an array job:
    // auto ioFiles = getFileIO();     // Load file names

    // Compute derived parameters
    double Asq = pow(params.at("A_0"), 2);
    double Ksq = (Asq * params.at("lam")) / (params.at("sA") * params.at("pInteract") * pow(params.at("a0"), 2.0));
    double Ag_avg = params["Ag_avg"];
    double gamma = params.at("n") * params.at("mu_A");
    params["Asq"] = Asq;
    params["Ksq"] = Ksq;
    params["Ag_avg"] = Ag_avg;
    params["gamma"] = gamma;

    // Scale number of T cells, thymic production rate, and infectivity according to order of magnitude
    double order_scale = pow(10, 11 - params.at("order_fullsim"));
    params["initial_T"] = params.at("initial_T") / order_scale;
    params["lam_T"] = params.at("lam_T") / order_scale;
    params["beta_initial"] = params.at("beta_initial") * order_scale;
    params["beta_setpoint"] = params.at("beta_setpoint") * order_scale;
    params["beta_ART"] = params.at("beta_ART") * order_scale;


    /* INITIALIZATION */

    // Initialize sequences and defectives
    std::vector<Sequence> sequences;
    std::vector<Defective> defectives;
    std::map<std::string, double> totals;
    initialize_clones(sequences, defectives, totals, params);

    // Initialize T cells and infectivity 
    double beta = params.at("beta_initial");
    double T = params.at("initial_T");
    double lam_T = params.at("lam_T");
    double mu_T = params.at("mu_T");
    double dt = params.at("dt");

    // Record initial conditions
    record_totals(totals, ioFiles.at("totals_outfile"), "w");
    record_sequences_clones(TIME, sequences, ioFiles.at("sequences_outfile"), ioFiles.at("clones_outfile"), "w");
    record_defectives(TIME, defectives, ioFiles.at("defectives_outfile"), "w");

    // Random number generation
    size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::normal_distribution<double> gaussian(0.0, 1.0);


    /* SIMULATION */

    // Iterate over time
    int n_steps = static_cast<int>((params.at("tEnd") - TIME) / dt);
    for (int _step=0; _step<n_steps; _step++) {
        // Print some progress updates and information based on time

        // Update beta based on time
        #ifdef EARLY
            if (TIME <= params.at("t_setpoint")) { beta = params.at("beta_initial"); }
            else if (TIME > params.at("t_setpoint") && TIME <= params.at("t_ART")) { beta = params.at("beta_initial"); }
            else { beta = params.at("beta_ART"); }
        #elif FULL_ATI
            if (TIME <= params.at("t_setpoint")) { beta = params.at("beta_initial"); }
            else if (TIME > params.at("t_setpoint") && TIME <= params.at("t_ART")) { beta = params.at("beta_setpoint"); }
            else if (TIME > params.at("t_ART") && TIME <= 120.0) {beta = params.at("beta_ART");}
            else {beta = params.at("beta_setpoint");}
        #elif EARLY_ATI
            if (TIME <= params.at("t_setpoint")) { beta = params.at("beta_initial"); }
            else if (TIME > params.at("t_setpoint") && TIME <= params.at("t_ART")) { beta = params.at("beta_initial"); }
            else if (TIME > params.at("t_ART") && TIME <= 12.5) {beta = params.at("beta_ART");}
            else {beta = params.at("beta_setpoint");}
        #else
            if (TIME <= params.at("t_setpoint")) { beta = params.at("beta_initial"); }
            else if (TIME > params.at("t_setpoint") && TIME <= params.at("t_ART")) { beta = params.at("beta_setpoint"); }
            else { beta = params.at("beta_ART"); }
        #endif

        // Evolve clones
        evolve_clones(sequences, defectives, beta, T, dt);

        // Update time
        TIME += dt;
        totals["t"] = TIME;

        // Enable mutations when TIME reaches t_ART
        if ((TIME >= params.at("t_ART")) && (TIME < params.at("t_ART") + dt)) {
           for (auto &seq : sequences) {
               seq.allow_mutations();
           }
        }

        // Remove extinct clones and generate new ones
        update_clones(sequences, defectives, totals, params);

        // Update T cells
        double dT = ((lam_T - (mu_T*T))*dt) - totals["nEvents"] + (sqrt((lam_T + (mu_T*T))*dt) * gaussian(rng));
        T += dT;
        if (std::isnan(T) || T<params.at("min_size")) { T = 0; }
        totals["T"] = T;

        // Record totals
        #ifdef EARLY_ATI
        if (((_step+1) % static_cast<int>(params.at("record_every")) == 0) 
            || (_step < 3/dt)
            || ((_step - 1 > params.at("t_ART")/dt) && (_step < (params.at("t_ART") + 6)/dt))
            || ((_step - 1 > (params.at("t_ART")+12)/dt) && (_step < ((params.at("t_ART")+12) + 6)/dt))
            ) {
            record_totals(totals, ioFiles.at("totals_outfile"), "a");
        }
        #elif FULL_ATI
        if (((_step+1) % static_cast<int>(params.at("record_every")) == 0) 
            || (_step < 3/dt)
            || ((_step - 1 > params.at("t_ART")/dt) && (_step < (params.at("t_ART") + 6)/dt))
            || ((_step - 1 > (params.at("t_ART")+60)/dt) && (_step < ((params.at("t_ART")+60) + 6)/dt))
            ) {
            record_totals(totals, ioFiles.at("totals_outfile"), "a");
        }
        #else
        if (((_step+1) % static_cast<int>(params.at("record_every")) == 0) 
            || (_step < 3/dt)
            || ((_step - 1 > params.at("t_ART")/dt) && (_step < (params.at("t_ART") + 6)/dt))
            ) {
            record_totals(totals, ioFiles.at("totals_outfile"), "a");
        }
        #endif

        // Record sequences and clones
        #ifdef EARLY_ATI
        if ((_step == 100-1) || (_step == 200-1) || (_step == 300-1) || (_step == 400-1) 
            || (_step == 6000-1) || (_step == 6400-1) || (_step == 6800-1) || (_step == 7200-1)
            || (_step == 440-1) || (_step == 540-1) || (_step == 1040-1) || (_step == 1640-1) || (_step == 6440-1)//added for early ATI
            ){ 
            //comment out during ATI studies:
            record_sequences_clones(TIME, sequences, ioFiles.at("sequences_outfile"), ioFiles.at("clones_outfile"), "a");
            record_defectives(TIME, defectives, ioFiles.at("defectives_outfile"), "a");
        }
        #elif FULL_ATI
        if ((_step == 100-1) || (_step == 200-1) || (_step == 300-1) || (_step == 400-1) 
            || (_step == 6000-1) || (_step == 6400-1) || (_step == 6800-1) || (_step == 7200-1)
            || (_step == 7300-1) || (_step == 7800-1) || (_step == 8400-1) || (_step == 13200-1) //added for regular ATI
            ){ 
            //comment out during ATI studies:
            record_sequences_clones(TIME, sequences, ioFiles.at("sequences_outfile"), ioFiles.at("clones_outfile"), "a");
            record_defectives(TIME, defectives, ioFiles.at("defectives_outfile"), "a");
        }
        #else
        if ((_step == 100-1) || (_step == 200-1) || (_step == 300-1) || (_step == 400-1) || (_step == 1200-1) || (_step == 2400-1)
           || (_step == 3000-1) || (_step == 3600-1) || (_step == 4800-1) || (_step == 6100-1) || (_step == 6200-1) || (_step == 6300-1)
           || (_step == 6000-1) || (_step == 6400-1) || (_step == 6800-1) || (_step == 7200-1) || (_step == 9600-1) || (_step == 8400-1) 
           || (_step == 10800-1) || (_step == 12000-1)
           ){ 
            // comment out during ATI studies:
           record_sequences_clones(TIME, sequences, ioFiles.at("sequences_outfile"), ioFiles.at("clones_outfile"), "a");
           record_defectives(TIME, defectives, ioFiles.at("defectives_outfile"), "a");
        }
        #endif
        
    }

    return EXIT_SUCCESS;

}
