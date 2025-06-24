#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include <map>
#include <vector>
#include <math.h>
#include <string>
#include <sstream>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <random>
#include <chrono>


// SEQUENCE CLASSES

// Latent clone class to store size, reactivation rate, and concentration of antigen
class LatentClone {
    public:
        // identifiers
        int id;           // clone ID (ID_CL)
        double t_created; // time created (ID_created)

        // dynamical variables
        double L;       // size of the clone
        double r;       // reactivation rate
        double Ag;      // antigen concentration
        double local_t; // local time for small clones
        double reactivated;

        // Check if sufficient clones exist
        bool is_extinct() const {
            return (L<0.5);  // min_size HARDCODED FOR NOW
        }

        LatentClone(int id, double t_created, double L, double r, double Ag, double reactivated, double local_t) 
            : id(id), t_created(t_created), L(L), r(r), Ag(Ag), reactivated(reactivated), local_t(local_t) {}
        ~LatentClone() {}
};

// Sequence class for replication-competent clones, actives, and virions
class Sequence {

    private:
        // new infection probabilities
        double p_def;    // probability defective
        double p_mut;    // probability of mutation
        double p_latent; // probability of latency
        bool no_mut;     // no mutation flag

        // replication parameters
        double nu_L;     // division rate of latent cells
        double mu_L;     // death rate of latent cells
        double mu_A;     // death rate of active cells
        double gamma;    // virion production rate
        double c;        // virion clearance rate

        // antigenic noise parameters
        double lam;      // antigenic noise rate
        double Asq;      // variance of antigenic noise
        double Ag_avg;   // average background antigen concentration

        // discrete/continuous parameters
        double min_size; // minimum size of clones

        // reactivation distribution parameters
        double mean_pR;  // mean reactivation probability
        double std_pR;   // standard deviation of reactivation probability

        // random number generation
        std::mt19937 rng{ std::random_device{} () };
        std::normal_distribution<double> gaussian;
        std::uniform_real_distribution<double> roll;


        // Evolve latent clones
        void evolve_latent(double dt) {

            double temp_react = 0;
            double L_REAC = 0;
            double L_STIM = 0;
            double L_PROF = 0;
            double L_DEATH = 0;

            double BIN1 = 0;
            double BIN2 = 0;
            double BIN3 = 0;
            double BIN4 = 0;
            double BIN5 = 0;

            // Note: antigen stimulation dt MISSING IN MARCO'S CODE, line 165 evolution.cpp

            // Iterate through latent clones
            for (auto &clone : clones) {

                /* SPLIT DYNAMICS */

                clone.reactivated = 0; //set by GK
                // Discrete for small clones
                if (clone.L<100) { //originally if clone.L < 100 (changed by GK)

                    double Ag_eff = Ag_avg + clone.Ag;      // effective antigen concentration
                    double r_tot = Ag_eff + nu_L + mu_L;    // total rate of events
                    std::discrete_distribution<> event_dist({Ag_eff, nu_L, mu_L});
                    while (clone.local_t<dt && clone.L>0) {

                        // Get time until next event
                        std::exponential_distribution<double> t_dist(r_tot*clone.L);
                        clone.local_t += t_dist(rng);

                        // Get event type
                        int event = event_dist(rng);
                        if (event==0) {                 // antigenic stimulation
                            if (roll(rng)<clone.r) {    // reactivation
                                temp_react += 1;        // clone reactivates
                                clone.reactivated = 1; //set clone reactivated state to true

                                if(clone.L < 10){
                                    BIN1+=1;
                                }
                                else if(clone.L>=10 && clone.L < 100){
                                    BIN2+=1;
                                }
                                else if(clone.L>=100 && clone.L < 1000){
                                    BIN3+=1;
                                }
                                else if(clone.L>=1000 && clone.L < 10000){
                                    BIN4+=1;
                                }
                                else{
                                    BIN5+=1;
                                }
                                
                                clone.L -= 1;           // remove one from the latent clone
                                L_REAC +=1;             // add one to reactivated latent cell count
                            } else {                    // division without reactivation
                                clone.L += 1;           // clone divides
                                L_STIM +=1;              // add one to replicated via Ag stimulation latent cell count
                            }
                        } else if (event==1) {          // division     
                            clone.L += 1;               // clone divides
                            L_PROF +=1;                  // add one to replicated latent cell count
                        } else {                        // death
                            clone.L -= 1;               // clone dies
                            L_DEATH +=1;                // add one to latent cell that dies
                        }

                    }

                    // Update local time for the next step
                    clone.local_t -= dt;

                }

                // Continuous for large clones
                else {

                    // Get growth stimulated by antigen and number that react 
                    double Ag_eff = Ag_avg + clone.Ag;
                    double L_Ag = (Ag_eff * clone.L * dt) + (sqrt(Ag_eff * clone.L * dt) * gaussian(rng));
                    L_Ag = stochastic_round(L_Ag);

                    double L_react = 0;
                    if (L_Ag>0) {
                        std::binomial_distribution<int> bin(L_Ag, clone.r);
                        L_react = bin(rng);
                    }
                    else L_Ag = 0;
                    L_REAC += L_react; //add to count of latent cells that reactivate
                    L_STIM += L_Ag; //add to count of latent cells that are stimulated by Ag

                    temp_react += L_react;

                    // Get birth/death
                    double L_bd = ((nu_L - mu_L) * clone.L * dt) + (sqrt((nu_L + mu_L) * clone.L * dt) * gaussian(rng));
                    //separating the terms to account for birth and death:
                    double L_b = (nu_L * clone.L * dt) + (sqrt((nu_L) * clone.L * dt) * gaussian(rng));
                    double L_d = (mu_L * clone.L * dt) + (sqrt((mu_L) * clone.L * dt) * gaussian(rng));

                    // Get net change
                    clone.L += L_bd + L_Ag - 2*L_react;
                    if(L_react>0){
                        clone.reactivated = 1;
                        
                        if(clone.L < 10){
                            BIN1+=1;
                        }
                        else if(clone.L>=10 && clone.L < 100){
                            BIN2+=1;
                        }
                        else if(clone.L>=100 && clone.L < 1000){
                            BIN3+=1;
                        }
                        else if(clone.L>=1000 && clone.L < 10000){
                            BIN4+=1;
                        }
                        else{
                            BIN5+=1;
                        }
                        
                    }
                    L_DEATH += L_d;
                    L_PROF += L_b;

                    // Round if small
                    if (clone.L<100) { clone.L = stochastic_round(clone.L); } //commented out by GK

                }

            }


            // Erase clones that are extinct efficiently
            clones.erase(std::remove_if(clones.begin(), clones.end(), [](const LatentClone &clone) {
                return clone.is_extinct();
            }), clones.end());

            n_react = temp_react;

            L_reac = L_REAC;
            L_stim = L_STIM;
            L_prof = L_PROF;
            L_death = L_DEATH;

            bin1 = BIN1;
            bin2 = BIN2;
            bin3 = BIN3;
            bin4 = BIN4;
            bin5 = BIN5;

        }

        // Evolve antigen concentrations 
        void evolve_antigen(double dt) {

            // Iterate through antigens
            for (auto &clone : clones) {

                double dAg = sqrt(2 * lam * Asq * dt) * gaussian(rng);
                clone.Ag += (-lam * clone.Ag * dt) + dAg;   // Dynamics of antigenic noise
            
            }

        }

        // Generate new infections 
        void new_infections(double beta, double T, double dt) {

            // Get number of new infections
            n_infect = (beta * T * V * dt) + (sqrt(beta * T * V * dt) * gaussian(rng));
            n_infect = stochastic_round(n_infect);

            // Sanity checks
            if (n_infect<0) n_infect = 0;
            if (n_infect>V) n_infect = V;

            // T cells shouldn't be exhausted, raise warning
            if (n_infect>T) {
                std::cout << "Unexpected: n_infect > T \n";
                std::cout << "n_infect = " << n_infect << "\n";
                std::cout << "T = " << T << "\n";

                n_infect = T;
            }

            // Assign fates - active vs. latent
            std::binomial_distribution<int> bin_al(n_infect, 1 - p_latent);
            double n_active = bin_al(rng);
            double n_latent = n_infect - n_active;

            // Assign fates - defective
            std::binomial_distribution<int> bin_def(n_latent, p_def);
            n_D = bin_def(rng);
            n_latent -= n_D;

            if (no_mut) {
                n_Am = 0;
                n_Lm = 0;
            } 
            
            else {
                std::binomial_distribution<int> bin_la(n_active, p_mut);
                std::binomial_distribution<int> bin_lm(n_latent, p_mut);
                n_Am = bin_la(rng);
                n_active -= n_Am;
                n_Lm = bin_lm(rng);
                n_latent -= n_Lm;
            }
            
            // Upate unmutated active and latent counts
            n_A = n_active;
            n_L = n_latent;
        
        }

        // Evolve actives and virions 
        void evolve_active(double dt) {

            // Get active dynamics
            double dA = n_react + n_A - (mu_A * A * dt) + (sqrt(mu_A * A * dt) * gaussian(rng));

            // Get virion dynamics
            double dV = (gamma * A * dt) - (c * V * dt) - n_infect + (sqrt(((gamma * A) + (c * V)) * dt) * gaussian(rng));

            // Update active and virion counts
            A += dA;
            V += dV;

            // Threshold active and virion counts
            if (A<min_size) A = 0;
            if (V<min_size) V = 0;

        }

        // Create new latent clones
        void create_new_latent_clones(double t) {

            // Distributions for probability of reactivation and antigen concentration
            std::normal_distribution<double> p_R(mean_pR, std_pR); //added by GK
            std::normal_distribution<double> Ag0(0, sqrt(Asq));

            // Create new latent clones
            for (int i=0; i<n_L; i++) {

                // Roll probability of reactivation
                double p_r = 0;
                do {
                    p_r = p_R(rng); //added by GK
                } while (p_r < -4.0 || p_r >= 0); //adjusted from -4.5 to -4.0 by GK

                #ifdef CONST_AG
                    clones.push_back(LatentClone(id_counter++, t, 1, pow(10, p_r), 0.002, false, 0));
                #elif CONST_PR_AG
                    clones.push_back(LatentClone(id_counter++, t, 1, pow(10, p_r), 0.002, false, 0));
                #else
                    clones.push_back(LatentClone(id_counter++, t, 1, pow(10, p_r), Ag0(rng), false, 0));
                #endif

            }

        }
    
    public:
        // identifiers
        double t_init;                  // time of initialization (ID_init)
        int id_sequence;                // overall sequence ID (ID_sequence)
        int id_counter;                 // counter for clone ID

        // dynamical variables
        double L_tot;                    // total number latent
        double A;                        // total number active
        double V;                        // number of virions
        std::vector<LatentClone> clones; // latent clones

        // statistics and new clone countsL_rep
        int Nmut;                   // number of mutations
        double LXR;                 // product of number latent and reactivation probability
        double n_react;             // total number of reactivations
        double n_infect;            // total number of infections
        int n_L;                    // NEW latent clones w/o mutations
        int n_A;                    // NEW actives w/o mutations 
        int n_Lm;                   // NEW latent clones w/  mutations
        int n_Am;                   // NEW actives w/  mutations
        int n_D;                    // NEW defective clones

        //added by GK:
        // double L_rep;
        double L_stim;
        double L_prof;
        double L_death;
        double L_reac;

        double bin1;
        double bin2;
        double bin3;
        double bin4;
        double bin5;



        // RNG accessory function
        double stochastic_round(double x) {
            if (roll(rng)>(x - floor(x))) return floor(x)+1;
            else return floor(x);
        }

        // Adjust mutation rate
        void allow_mutations() {
            no_mut = false;
        }

        // Check if sufficient clones exist
        bool is_extinct() const {
            return (L_tot<min_size && A<min_size && V<min_size);
        }

        // Update and check if sequence is extinct
        bool update_extinct(std::map<std::string, double> &totals, 
                            std::map<int, int> &new_latent,
                            std::map<int, int> &new_active,
                            int &new_defective) const {

            // Check if sequence is extinct
            if (is_extinct()) { return true; }
           
            // Else update totals and return false
            else {

                totals["Ltotal"] += L_tot;
                totals["Atotal"] += A;
                totals["Vtotal"] += V;
                totals["LXRtotal"] += LXR;
                totals["nSequences"] += 1;
                totals["nLclones"] += clones.size();
                if (Nmut>0) {
                    totals["nMclones"] += clones.size();
                    totals["Lmut"] += L_tot;
                    totals["Amut"] += A;
                    totals["Vmut"] += V;
                }
                totals["nL"] += n_L;
                totals["nmL"] += n_Lm;
                totals["nmA"] += n_Am;
                totals["nD"] += n_D;
                totals["nEvents"] += n_infect;
                totals["nReact"] += n_react;
                
                totals["Lstim"] += L_stim;
                totals["Lprof"] += L_prof;
                totals["Ldeath"] += L_death; 
                totals["Lreac"] += L_reac;

                totals["bin1"] += bin1;
                totals["bin2"] += bin2;
                totals["bin3"] += bin3;
                totals["bin4"] += bin4;
                totals["bin5"] += bin5;

                // Update new clone counts based on mutations
                if (n_Lm>0) {
                    if (new_latent.contains(Nmut+1)) new_latent[Nmut+1] += n_Lm;
                    else new_latent[Nmut+1] = n_Lm;
                }
                if (n_Am>0) {
                    if (new_active.contains(Nmut+1)) new_active[Nmut+1] += n_Am;
                    else new_active[Nmut+1] = n_Am;
                }
                new_defective += n_D;

                return false;

            }

        }

        // Externally create n new latent clones
        void add_new_latent(int n, double t) {

            int temp_n_L = n_L;
            n_L = n; // Set number of new latent clones
            create_new_latent_clones(t);
            n_L = temp_n_L;

        }
        
        // Evolve all components 
        void evolve(double beta, double T, double dt, double t) {

            #ifdef CONST_AG

                // Evolve latent clones
                evolve_latent(dt);

                // Generate new infections
                new_infections(beta, T, dt);

                // Evolve active cells and virions
                evolve_active(dt);

                // Create new latent clones
                create_new_latent_clones(t);

                // Update L_tot and LXR
                L_tot = 0;
                LXR = 0;
                for (auto &clone : clones) {
                    L_tot += clone.L;
                    LXR += clone.L * clone.r;
                }

            #elif CONST_PR_AG

                // Evolve latent clones
                evolve_latent(dt);

                // Generate new infections
                new_infections(beta, T, dt);

                // Evolve active cells and virions
                evolve_active(dt);

                // Create new latent clones
                create_new_latent_clones(t);

                // Update L_tot and LXR
                L_tot = 0;
                LXR = 0;
                for (auto &clone : clones) {
                    L_tot += clone.L;
                    LXR += clone.L * clone.r;
                }

            #else

                // Evolve latent clones
                evolve_latent(dt);

                // Evolve antigens
                evolve_antigen(dt); //comment out when running constant antigen simulation

                // Generate new infections
                new_infections(beta, T, dt);

                // Evolve active cells and virions
                evolve_active(dt);

                // Create new latent clones
                create_new_latent_clones(t);

                // Update L_tot and LXR
                L_tot = 0;
                LXR = 0;
                for (auto &clone : clones) {
                    L_tot += clone.L;
                    LXR += clone.L * clone.r;
                }

            #endif

        }

        Sequence() {}
        Sequence(std::size_t seed, const std::map<std::string, double> &parameters, double t, int id) 
            : rng(seed), roll(0.0, 1.0), gaussian(0.0, 1.0) {
            
            // Set parameters from the map
            p_def = parameters.at("p_def");
            p_mut = parameters.at("p_mut");
            p_latent = parameters.at("p_latent");
            nu_L = parameters.at("nu_L");
            mu_L = parameters.at("mu_L");
            mu_A = parameters.at("mu_A");
            gamma = parameters.at("gamma");
            c = parameters.at("c");
            lam = parameters.at("lam");
            Asq = parameters.at("Asq");
            Ag_avg = parameters.at("Ag_avg");
            min_size = parameters.at("min_size");
            mean_pR = parameters.at("mean_pR");
            std_pR = parameters.at("std_pR");

            // Initialize other variables
            no_mut = true;      // no mutations by default
            t_init = t;
            id_sequence = id;
            id_counter = 0;
            L_tot = 0;
            A = 0;
            V = 0;
            Nmut = 0;
            LXR = 0;
            n_react = 0;
            n_infect = 0;
            n_L = 0;
            n_A = 0;
            n_Lm = 0;
            n_Am = 0;
            n_D = 0;
            // L_rep = 0;
            L_stim = 0;
            L_prof = 0;
            L_death = 0;
            L_reac = 0;

            bin1 = 0;
            bin2 = 0;
            bin3 = 0;
            bin4 = 0;
            bin5 = 0;

        }
        ~Sequence() {}
	
};


// Sequence class for replication-incompetent clones (defectives)
class Defective {

    private:
        // replication parameters
        double nu_L;     // division rate of latent cells
        double mu_L;     // death rate of latent cells

        // antigenic noise parameters
        double lam;      // antigenic noise rate
        double Asq;      // variance of antigenic noise
        double Ag_avg;   // average background antigen concentration

        // discrete/continuous parameters
        double min_size; // minimum size of clones

        // ^ Future: allow for rare "reactivation", death through expression of viral proteins, etc.

        // random number generation
        std::mt19937 rng{ std::random_device{} () };
        std::uniform_real_distribution<double> roll;
        std::normal_distribution<double> gaussian;

        // Evolve defective clones
        void evolve_defective(double dt) {

            /* SPLIT DYNAMICS */

            // Discrete for small clones
            if (D<100) {

                // Get effective antigen concentration
                double Ag_eff = Ag_avg + Ag;
                double r_tot = Ag_eff + nu_L + mu_L;
                double t_evolve = 0;
                std::discrete_distribution<> event_dist({Ag_eff + nu_L, mu_L});
                while (t_evolve<dt && D>0) {

                    // Get time until next event
                    std::exponential_distribution<double> t_dist(r_tot*D);
                    t_evolve += t_dist(rng);

                    // Get event type
                    int event = event_dist(rng);
                    if (event==0) { D += 1; }   // clone divides
                    else { D -= 1; }            // clone dies

                }
            
            }

            // Continuous for large clones
            else { 

                // Get growth stimulated by antigen
                double Ag_eff = Ag_avg + Ag;
                double D_Ag = 0;
                D_Ag = (Ag_eff * D * dt) + (sqrt(Ag_eff * D * dt) * gaussian(rng));
                if (D_Ag<0) D_Ag = 0;

                // Get birth/death
                double D_bd = ((nu_L - mu_L) * D * dt) + (sqrt((nu_L + mu_L) * D * dt) * gaussian(rng));

                // Get net change
                D += D_bd + D_Ag;

                // Round if small
                if (D<100) D = stochastic_round(D);

                // if (D<min_size) D=0;

            }

        }

        // Evolve antigen concentrations 
        void evolve_antigen(double dt) {

            // Iterate through antigens
            double dAg = sqrt(2 * lam * Asq * dt) * gaussian(rng);
            Ag += (-lam * Ag * dt) + dAg;   // Dynamics of antigenic noise
            
        }
    
    public:
        double t_created;   // time created (ID_created)
        int id_sequence;    // sequence ID (ID_sequence)
        double D;           // total number latent defective
        double Ag;          // fluctuating antigen concentration


        // RNG accessory function
        double stochastic_round(double x) {
            if (roll(rng)>(x - floor(x))) return floor(x)+1;
            else return floor(x);
        }

        // Check if sufficient clones exist
        bool is_extinct() const {
            return (D<min_size);
        }

        // Update and check if sequence is extinct
        bool update_extinct(std::map<std::string, double> &totals) const {
        
            // Check if sequence is extinct
            if (is_extinct()) { return true; }

            // Else update totals and return false
            else {
                totals["Dtotal"] += D;
                totals["nDclones"] += 1;
                return false;
            }

        }
        
        // Evolve all components 
        void evolve(double dt) {

            #ifdef CONST_AG
                // Evolve defective clones
                evolve_defective(dt);
            #elif CONST_PR_AG
                // Evolve defective clones
                evolve_defective(dt);
            #else
                // Evolve defective clones
                evolve_defective(dt);

                // Evolve antigens
                evolve_antigen(dt); //comment out if doing constant Ag simulations
            #endif

        }

        Defective() {}
        Defective(std::size_t seed, const std::map<std::string, double> &parameters, double t, int id) 
            : rng(seed), roll(0.0, 1.0), gaussian(0.0, 1.0) {
            
            // Set parameters from the map
            nu_L = parameters.at("nu_L");
            mu_L = parameters.at("mu_L");
            lam = parameters.at("lam");
            Asq = parameters.at("Asq");
            Ag_avg = parameters.at("Ag_avg");
            min_size = parameters.at("min_size");

            // Initialize other variables
            std::normal_distribution<double> Ag0(0, sqrt(Asq));
            t_created = t;
            id_sequence = id;
            D = 1;
            #ifdef CONST_AG
                Ag = 0.002;
            #elif CONST_PR_AG
                Ag = 0.002;
            #else
                Ag = Ag0(rng);
            #endif

        }
        ~Defective() {}
	
};


#endif
