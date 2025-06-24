#ifndef CONFIG_H
#define CONFIG_H

#include <cmath>
#include <map>
#include <string>

// Return a dictionary for file input/output
inline std::map<std::string, std::string> getFileIO(int taskID) {
    //int taskID (pass this into the method when running an array job)
    return {
        // Input files for initial conditions
        {"sequences_infile", "input/full_simulation/sequences_0.0.csv"},
        {"clones_infile", "input/full_simulation/clones_0.0.csv"},
        {"defectives_infile", "input/full_simulation/defectives_0.0.csv"},

        #ifdef FULL
            {"sequences_outfile", "output/final_runs/full_simulation_ord9/sequences_"+std::to_string(taskID)+".csv"},
            {"clones_outfile", "output/final_runs/full_simulation_ord9/clones_"+std::to_string(taskID)+".csv"},
            {"defectives_outfile", "output/final_runs/full_simulation_ord9/defectives_"+std::to_string(taskID)+".csv"},
            {"totals_outfile", "output/final_runs/full_simulation_ord9/totals_"+std::to_string(taskID)+".csv"},
        #elif EARLY
            {"sequences_outfile", "output/final_runs/full_simulation_ord9_early/run"+std::to_string(taskID)+"/sequences.csv"},
            {"clones_outfile", "output/final_runs/full_simulation_ord9_early/run"+std::to_string(taskID)+"/clones.csv"},
            {"defectives_outfile", "output/final_runs/full_simulation_ord9_early/run"+std::to_string(taskID)+"/defectives.csv"},
            {"totals_outfile", "output/final_runs/full_simulation_ord9_early/run"+std::to_string(taskID)+"/totals.csv"},
        #elif FULL_ATI
            {"sequences_outfile", "output/final_runs/full_ATI/sequences_"+std::to_string(taskID)+".csv"},
            {"clones_outfile", "output/final_runs/full_ATI/clones_"+std::to_string(taskID)+".csv"},
            {"defectives_outfile", "output/final_runs/full_ATI/defectives_"+std::to_string(taskID)+".csv"},
            {"totals_outfile", "output/final_runs/full_ATI/totals_"+std::to_string(taskID)+".csv"},
        #elif EARLY_ATI
            {"sequences_outfile", "output/final_runs/early_ATI/sequences_"+std::to_string(taskID)+".csv"},
            {"clones_outfile", "output/final_runs/early_ATI/clones_"+std::to_string(taskID)+".csv"},
            {"defectives_outfile", "output/final_runs/early_ATI/defectives_"+std::to_string(taskID)+".csv"},
            {"totals_outfile", "output/final_runs/early_ATI/totals_"+std::to_string(taskID)+".csv"},
        #elif CONST_PR
            {"sequences_outfile", "output/final_runs/full_simulation_ord9_constant_pR/sequences.csv"},
            {"clones_outfile", "output/final_runs/full_simulation_ord9_constant_pR/clones.csv"},
            {"defectives_outfile", "output/final_runs/full_simulation_ord9_constant_pR/defectives.csv"},
            {"totals_outfile", "output/final_runs/full_simulation_ord9_constant_pR/totals.csv"},
        #elif CONST_AG
            {"sequences_outfile", "output/final_runs/full_simulation_ord9_constant_Ag/sequences.csv"},
            {"clones_outfile", "output/final_runs/full_simulation_ord9_constant_Ag/clones.csv"},
            {"defectives_outfile", "output/final_runs/full_simulation_ord9_constant_Ag/defectives.csv"},
            {"totals_outfile", "output/final_runs/full_simulation_ord9_constant_Ag/totals.csv"},
        #elif CONST_PR_AG
            {"sequences_outfile", "output/final_runs/full_simulation_ord9_constant_pR_Ag/sequences.csv"},
            {"clones_outfile", "output/final_runs/full_simulation_ord9_constant_pR_Ag/clones.csv"},
            {"defectives_outfile", "output/final_runs/full_simulation_ord9_constant_pR_Ag/defectives.csv"},
            {"totals_outfile", "output/final_runs/full_simulation_ord9_constant_pR_Ag/totals.csv"},
        #endif
    };
}

// Return a dictionary of parameter values
inline std::map<std::string, double> getParameters() {
    return {
        // All rates are in month^-1.

        // Minimum size threshold for clones
        {"min_size", 0.5},                      // minimum size threshold for clones

        // Recording parameters
        {"record_every", 100.0},                // record every 100 time steps
    
        // Orders of magnitude where the simulations will run
        {"order_fullsim", 9.0},                // order of magnitude (number of T cells)
        {"orderL_fullsim", 9.0},               // order of magnitude (number of latent cells)

        // Probabilities during infection events
        {"p_latent", 0.05},                     // probability of latent infection
        {"p_def", 0.002},                       // probability of defective infection
        {"p_mut", 0.33},                        // probability of mutation
        
        // Latent parameters
        {"nu_L", 29.4},   // 29.4 = 0.98*30     // latent division rate (29.4 originally) //originally 29.4
        {"mu_L", 44.137}, // 29.4 + 14.712      // latent death rate (originally 44.112) (44.132 in last iteration) //originally 44.137 (GK)
        
        // Distribution of probabilities of reactivation
        #ifdef CONST_PR
            {"mean_pR", -2.0},
            {"std_pR", 0.0},
        #elif CONST_PR_AG
            {"mean_pR", -2.0},
            {"std_pR", 0.0},
        #else
            {"mean_pR", -1.0},
            {"std_pR", 0.8},
        #endif

        // Antigen parameters -- yields Ag_avg ~ 15              
        {"lam", 3.0},  // 0.1*30                // antigenic noise rate
        {"sA", 58.8e6}, // 30 * 1.96e6          // THESE ARE DIFFERENT FROM THOSE IN THE PAPER 
        {"pInteract", 1.0e-7},
        {"a0", 1.0},
        {"A_0", 0.3}, //originally 105*10**-8 (GK)
        {"Ag_avg", 14.7}, //originally 14.702 //originally 14.7

        // Active and virion parameters
        {"mu_A", 21.0},                         // active cell death rate
        {"n", 5000.0},                          // virion burst sizeplot_lr_distribution_early
        {"c", 150.0},                           // virion clearance rate
        
        // Infectivities
        {"beta_initial", 4e-13},
        {"beta_setpoint", 2.76e-13}, // 1.70 was too low watch out for oscillatory dynamics! (originally 2.76e-13)
        {"beta_ART", 60e-15},
        
        // T cell parameters
        {"lam_T", 1.05e10}, // 70*5000*30*1000  // Thymic production rate
        {"mu_T", 0.06},                         // T cell death rate
        {"initial_T", 1.75e11},                 // Initial number of T cells

        // Time dynamics (in months)
        {"dt", 0.01},                           // time step for simulation
        {"dtt", 0.0001},

        // Times when events occur
        #ifdef EARLY
            {"t_ART", 0.5},
            {"t_setpoint", -1.00}, // We shift from initial beta to beta during chronic infection
            {"tEnd", 800.0},
        #elif EARLY_ATI
            {"t_ART", 0.5},
            {"t_setpoint", -1.00}, // We shift from initial beta to beta during chronic infection
            {"tEnd", 800.0},
        #else
            {"t_ART", 60},
            {"t_setpoint", 1.00}, // We shift from initial beta to beta during chronic infection
            {"tEnd", 800.0},
        #endif
        
        // Multiprocessing config
        {"nthre", 12.0} // Use 12 threads in mac, 128 threads in cluster. Adjust depending on compute available
    };
}

#endif // CONFIG_H

