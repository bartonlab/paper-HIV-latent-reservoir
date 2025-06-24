#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <vector>
#include <math.h>
#include <iostream>
#include <sstream> // std::stringstream
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
//#include "tools.h"  // Numerical tools
#include <chrono>
#include <omp.h>

//#include <Eigen/Dense>

#include <thread>

// Note: parameters have been converted to their equivalent in months, they are not in days

float order  = 7.0;   //normal is 11
float orderL = 7.0;
float monthsART = 60.00; //months before ART starts
//float monthsART = 0.03;
// T cells parameters

float T = 1.75*(pow(10,order));
//float T = 16187047.0;

float lam_T = 1.05*(pow(10,order-1)); //70*5000*30*1000
float mu_T  = 0.002*30; // 0.06

float beta  = 4.0*(pow(10,-order-2));

//float beta  = 18.0*(pow(10,-order-3));


float idmut = monthsART*100;
float dt = 0.01;
float tEnd = 800.00; //tend 800 should be
//float tEnd = 24.00;
//float tEnd = 92.00;
//float tEnd = 80.00*10;

// // Function to read csv files // //
std::vector<std::pair<std::string, std::vector<float> > > read_csv(std::string filename){
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<std::pair<std::string, std::vector<float> > > result;
    // Create an input filestream
    std::ifstream myFile(filename);
    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");
    // Helper vars
    std::string line, colname;
    float val;
    // Read the column names
    if(myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);
        // Create a stringstream from line
        std::stringstream ss(line);
        // Extract each column name
        while(std::getline(ss, colname, ',')){
            // Initialize and add <colname, int vector> pairs to result
            result.push_back(std::make_pair(colname,std::vector<float> () ));
        }
    }
    // Read data, line by line
    while(std::getline(myFile, line))
    {
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

// // Function to evolve sequences // //
void evolve_sequences(float &L_sequence, float &A_sequence, float &V_sequence, float &LXR_sequence, float &Nclones_sequence, float &nLs_sequence, float &nmLs_sequence, float &nmAs_sequence, float &nDs_sequence, float &nEventss_sequence, float &nReact_sequence, float &dt_var, int &dt_action, int &dt_lat, float beta, float T, std::vector<float> &Rclones, std::vector<float> &Lclones, std::vector<float> &AGclones, std::vector<int> &IDclones, int &COUNTERclones, int threadnum1, float nEventss){

    // lat is index of the latent clone that a cell will divide or die or whatever
    int lat = dt_lat;
    // Convert to local variables
    float Lsequence   = L_sequence;
    float Asequence   = A_sequence;
    float Vsequence   = V_sequence;
    float LXRsequence = LXR_sequence;
    float Nclones     = Nclones_sequence;
    float nLs         = 0;
    float nAs         = 0;
    float nmLs        = 0;
    float nmAs        = 0;
    float nDs         = 0;
//    float nEventss    = 0;
    int counter       = COUNTERclones;
    std::vector<float> REACTIVATION = Rclones;
    std::vector<float> LATENT       = Lclones;
    std::vector<float> ANTIGEN      = AGclones;
    std::vector<int> ID_CL          = IDclones;
    
//    std::cout << Vsequence << " V start of function \n";
    
    float order  = 7.0;
    float orderL = 7.0;
    
    //Latent parameters
    float nu_L = 0.00+29.4; //division rate of latent cells
    float mu_L = 7.195+29.4; //death rate of latent cells   36.595
    
    
    
    // Probabilities
    float p_L    = 0.05;  // In reality it is 0.06
    float p_def  = 0.002;
    float p_mut  = 0.33;
    
    
    float p_surv = 1/(pow(10,(orderL-order))); //exponent is how many orders of magnitude L is above A
    
    // Antigen parameters
    float lam       = 60.00;
    float sA        = 30*1.96*1.0e+7;
    float pInteract = 1.0e-7;
    float a0        = 1.0;
    float Ksq       = 1.0;
    float Asq       = sA*pInteract*(pow(a0,2.0))*Ksq/lam;
    
    
    // Note: The value gamma_f or the strength of variability of antigenic environment is equal to sqrt(lam * Asq) = A * sqrt(lam) in this case that is 7.6681158
    
    
    
    // Active and virion parameters
    float mu_A      = 21;
    float n         = 5000;
    float gammas    = n*mu_A;
    float c         = 150;
    // Time dynamics
    float dt  = 0.01;
    float dtt = 0.0001;
    // Global
    float idmut = monthsART*100;
    
    //Generators
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);      // Normal distribution

    std::normal_distribution<double> distribution_PR(-1.0,0.8);  //pR distribution normal
    
    std::normal_distribution<double> distribution3(0,sqrt(Asq)); // a0 initialization
    
    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count()+threadnum1*1000);
//    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    
    
    //Start of code
    float react = 0; //Number of cells that reactivate
    
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() + threadnum1*1000);
    //                std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::binomial_distribution<> dD(1, p_def);  //probability of defective
    std::binomial_distribution<> dM(1, p_mut);  //probability of mutation
    std::binomial_distribution<> dL(1, p_L);  //probability of latency
    std::binomial_distribution<> dSurv(1, p_surv); //probability of surviving after reactivating (needed because of running 1 orders below)
        
    int valueD = 0;
    int valueM = 0;
    int valueL = 0;
    int valueR = 0;
    int valueS = 0;
    float Agtemp = 0;
    
    if (!(Asequence > 0 || Vsequence > 0 || Lsequence > 0)){
        std::cout<<"problem now \n";
    }
    
		
    dt_action = 0; //Reset the action so in case it falls to less than 200 there wont be a problem
    
    Lsequence   = 0;
    LXRsequence = 0;

    // Loop for the dynamics of the clones in sequence j (reversed to be able to erase clones that die)
    for (int k = Nclones-1; k>=0; k = k-1){
        
        
        //Updating number of cells that reactivate in the time step dt contributing to sequence j (done before L dynamics)
        react += (nu_L+ANTIGEN.at(k))*LATENT.at(k)*REACTIVATION.at(k);
//            react += (ANTIGEN.at(k))*LATENT.at(k)*REACTIVATION.at(k)*dt;
        
        //Dynamics of the latent clone
        LATENT.at(k) += ((1-2*REACTIVATION.at(k))*(nu_L + ANTIGEN.at(k)) - mu_L)*LATENT.at(k)*dt + sqrt((nu_L+ANTIGEN.at(k)+mu_L)*LATENT.at(k)*dt)*distribution(generator);
//            LATENT.at(k) += ((1-2*REACTIVATION.at(k))*(ANTIGEN.at(k)) + nu_L - mu_L)*LATENT.at(k)*dt + sqrt((nu_L+ANTIGEN.at(k)+mu_L)*LATENT.at(k)*dt)*distribution(generator);
        
        
        if (LATENT.at(k)<0.5){
        LATENT.at(k)=0;
        }
        
        // Loop for antigen dynamics (faster dynamics)
        for(int l=0; l<(dt/dtt); l++){
            Agtemp = sqrt(2. *lam *Asq *dtt) * distribution(generator);
            if (Agtemp<0){
                Agtemp=0;
            } //closing if for Agtemp<0
            ANTIGEN.at(k)  += (-lam * ANTIGEN.at(k) * dtt) + Agtemp;   // Dynamics of antigens
            if (ANTIGEN.at(k)<0){
                ANTIGEN.at(k)=0;
            } // closing if for Ag<0
        }
// closing loop over l (antigen dynamics)
    
        Lsequence   += LATENT.at(k);
        LXRsequence += LATENT.at(k)*REACTIVATION.at(k);
        
        // Delete clone if L = 0
        if (LATENT.at(k) == 0){
            ID_CL.erase(ID_CL.begin()+k);
            REACTIVATION.erase(REACTIVATION.begin()+k);
            LATENT.erase(LATENT.begin()+k);
            ANTIGEN.erase(ANTIGEN.begin()+k);
        }
    } // closing loop over k (clone k of sequence j)
    
    
    
    // This is because Latent is one order of magnitude above the rest and is better than just multiplying by p_surv
    float Nreac = react;
//    react = 0;
//    for( int is = 0; is < Nreac; is++) {
//        valueS = dSurv(gen);
//        if (valueS == 1){
//            react+=1;
//        }
//    }
//    std::cout << Vsequence << " V before if A>200 \n";

//    if (Asequence > 200){  this is where that used to be
//    nEventss = round(beta*T*Vsequence*dt);
    
    
    //Multinomial choice
    for(int k = 0; k < nEventss; k++) {
        valueD = dD(gen);
        if (valueD == 0){
            // not defective
            valueM = dM(gen);
            if (valueM == 0){
                // not mutation
                valueL = dL(gen);
                if (valueL == 0){
                    //not latent
                    nAs++;
                }
                else{
                    //latent
                    nLs++;
                }
            }
            else{
                // mutation
                valueL = dL(gen);
                if (valueL == 0){
                    //not latent
                    nmAs++;
                }
                else{
                    //latent
                    nmLs++;
                }
            }
        }
        else{
            // defective
            nDs++;
        }
    }
    
    
    Lsequence += nLs;
    
    
    // Calculating noises A and V
    float D11 = 0;
    float D12 = 0;
    float D21 = 0;
    float D22 = 0;
    
    
    if (Vsequence==0){
        D11 = sqrt(Asequence*mu_A + react) ;
        D12 = 0;
        D21 = 0;
        D22 = sqrt(Asequence*gammas);
    }
    else{
        float nAs2 = beta*T*Vsequence*(1-p_def)*(1-p_mut)*(1-p_L);
        float x1 = Asequence*(gammas + mu_A) + nAs2 + beta*T*Vsequence + react + c*Vsequence;
        float x2_1 = pow((gammas*Asequence + beta*T*Vsequence + c*Vsequence + react + mu_A*Asequence + nAs2),2);
        float x2_2 = -4*((react+mu_A*Asequence+nAs2)*(gammas*Asequence + beta*T*Vsequence+c*Vsequence) -pow(nAs2,2));
        float x2   = sqrt(x2_1 + x2_2);
        float x3   = beta*T*Vsequence + Asequence*(gammas-mu_A) - nAs2 - react + c*Vsequence;

        float eigval1 = (x1-x2)/2;
        float eigval2 = (x1+x2)/2;


        float V11 = (x3+x2)/(2*nAs2);
        float V12 = (x3-x2)/(2*nAs2);
        float V21 = 1;
        float V22 = 1;

        float I11 =  nAs2/x2;
        float I12 =  (-x3+x2)/(2*x2);
        float I21 = -nAs2/x2;
        float I22 =  (x3+x2)/(2*x2);

        D11 = sqrt(eigval1)*I11*V11 + sqrt(eigval2)*I21*V12;
        D12 = sqrt(eigval1)*I12*V11 + sqrt(eigval2)*I22*V12;
        D21 = sqrt(eigval1)*I11*V21 + sqrt(eigval2)*I21*V22;
        D22 = sqrt(eigval1)*I12*V21 + sqrt(eigval2)*I22*V22;
    }
    // // // // // finishes here
    
    
    float random1 = sqrt(dt)*distribution(generator);
    float random2 = sqrt(dt)*distribution(generator);
    
    float noise_A = D11*random1 + D12*random2;
    float noise_V = D21*random1 + D22*random2;
    //End of calculating noise
    
    // Dynamics of actives and virions
    Vsequence += gammas*Asequence*dt - nEventss - c*Vsequence*dt + noise_V;
    
//    Asequence += react;
    Asequence += nAs-mu_A*Asequence*dt + react*dt + noise_A;
    
//        if (Asequence != Asequence){
//            std::cout << "problem in A" << "\n";
//        }
    
//        if (Asequence<0.5){
//            Asequence = 0;
//        }
//        if (Vsequence<0.5){
//            Vsequence = 0;
//        }
//    std::cout << Vsequence << " V after if A>200 \n";

    
    if (Asequence<0.5){
        Asequence = 0;
    }
    if (Vsequence<0.5){
        Vsequence = 0;
    }
    
    
    //Create new latent clones without mutation
    float lpr = 0;
    float a00 = 0;
    for (int k=0; k<nLs*pow(10,(orderL-order)); k++){
        ID_CL.push_back(counter);
        counter++;
//        do{
//            lpr = -distribution2(generator);
//        }while(lpr<-3.2);
        
        do{
            lpr = distribution_PR(generator);
        }while(lpr<-4.0 or lpr>0);
        
        REACTIVATION.push_back(pow(10,lpr));
        LATENT.push_back(1);
        a00 = distribution3(generator);
        if (a00<0){
        a00 = 0;
        }
        ANTIGEN.push_back(a00);
        LXRsequence+=pow(10,lpr);
    }
    
    // // Updating values for sequence j // //
    dt_lat = lat;
    L_sequence        = Lsequence;
    A_sequence        = Asequence;
    V_sequence        = Vsequence;
    LXR_sequence      = LXRsequence;
    Nclones_sequence  = LATENT.size();
    nLs_sequence      = nLs;
    nmLs_sequence     = nmLs;
    nmAs_sequence     = nmAs;
    nDs_sequence      = nDs;
    nEventss_sequence = nEventss;
    nReact_sequence   = react*dt;
    Rclones           = REACTIVATION;
    Lclones           = LATENT;
    AGclones          = ANTIGEN;
    IDclones          = ID_CL;
    COUNTERclones     = counter;
}

// // Function to evolve defectives
void evolve_defectives(float &DEFECTIVES, float &ANTIGEN, int threadnum1){
    // Convert to local variables
    float D  = DEFECTIVES;
    float Ag = ANTIGEN;
     // Antigen parameters
    float lam = 60.00;
    float sA  = 30*1.96*1.0e+7;
    float pInteract = 1.0e-7;
    float a0 = 1.0;
    float Ksq = 1.0;
    float Asq = sA*pInteract*(pow(a0,2.0))*Ksq/lam;
    // Latent cells parameters
    float nu_L = 29.4; //division rate of latent cells
    float mu_L = 7.215+29.4; //death rate of latent cells
    float dt = 0.01;
    float dtt = 0.0001;
    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count()+threadnum1*1000);
    
//    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
//                std::cout << omp_get_thread_num() << "\n";
    float Agtemp = 0;
    
    D += (nu_L + Ag - mu_L)*D*dt + sqrt((nu_L+Ag+mu_L)*D*dt)*distribution(generator);
    if (D<0.5){
    D=0;
    }
    
    // Loop for antigen dynamics (faster dynamics)
    for(int l=0; l<(dt/dtt); l++){
        Agtemp = sqrt(2. *lam *Asq *dtt) * distribution(generator);
        if (Agtemp<0){
            Agtemp=0;
        } //closing if for Agtemp<0
    Ag  += (-lam * Ag * dtt) + Agtemp;   // Dynamics of antigens
    if (Ag<0){
        Ag=0;
    } // closing if for Ag<0
    } // closing loop over l (antigen dynamics)
    
    //Updating values for defectives j
    DEFECTIVES = D;
    ANTIGEN    = Ag;
}

int main(int argc, char *argv[]) {

//    // Process command line input
//
//    std::string output_file       = "out";
//    int         n_trials          = 100;
//    double      population_cutoff = 1e3;
//
//    for (int i=1;i<argc;i++) {
//
//        if      (strcmp(argv[i],"-o")==0) { if (++i==argc) break; else output_file       = argv[i];              }
//        else if (strcmp(argv[i],"-n")==0) { if (++i==argc) break; else n_trials          = strtoint(argv[i]);    }
//        else if (strcmp(argv[i],"-p")==0) { if (++i==argc) break; else population_cutoff = strtodouble(argv[i]); }
//
//        else printf("Unrecognized command! '%s'\n", argv[i]);
//
//    }
//
//    // Open output file
//
//    FILE *out = fopen((output_file+".csv").c_str(),"w");
//    fprintf(out,"trial,time,reactivation_rate,burst_rate,poisson_burst_size,death_rate,n,mutations\n");
    

    //Read the initial condition from the input files
    std::vector<std::pair<std::string, std::vector<float> > > SEQUENCES = read_csv("initial/sequences_0.0.csv");
    std::vector<std::pair<std::string, std::vector<float> > > CLONES = read_csv("initial/clones_0.0.csv");
    std::vector<std::pair<std::string, std::vector<float> > > DEFECTIVES = read_csv("initial/defectives_0.0.csv");

//	//Read the initial condition from the input files
//    std::vector<std::pair<std::string, std::vector<float> > > SEQUENCES = read_csv("initial/sequences_80.0.csv");
//    std::vector<std::pair<std::string, std::vector<float> > > CLONES = read_csv("initial/clones_80.0.csv");
//    std::vector<std::pair<std::string, std::vector<float> > > DEFECTIVES = read_csv("initial/defectives_80.0.csv");
    
    // Time when simulation starts
    float tStart = SEQUENCES.at(0).second.at(0);
    
    // Here I can add read the totals csv, get the latest value of T cells and use that for the initial value of T cells
    
    
    // Erase the vector of time
    SEQUENCES.erase(SEQUENCES.begin()+0);
    DEFECTIVES.erase(DEFECTIVES.begin()+0);
    // Erase vectors of time, ID_created, ID_sequence, ID_clone
    CLONES.erase(CLONES.begin()+0);
    CLONES.erase(CLONES.begin()+0);
    CLONES.erase(CLONES.begin()+0);
    
    // // Getting total values // //
    int nLclones   = 0;   // Number of latent clones
    int nMclones   = 0;   // Number of mutant clones
    int nDclones   = DEFECTIVES.at(0).second.size();
    int nSequences = SEQUENCES.at(0).second.size(); // Number of different seqs
    float Ltotal   = 0;   // Total number of latent cells
    float Atotal   = 0;   // Total number of active cells
    float Vtotal   = 0;   // Total number of virions
    float LXRtotal = 0;   // Accumulated sum of latent times reactivation
    float Lmut     = 0;   // Number of latent cells that are mutants
    float Amut     = 0;   // Number of active cells that are mutants
    float Vmut     = 0;   // Number of virions that are mutants
    float Dtotal   = 0;   // Total number of defectives
    int nEvents    = 0; // Total number of infecting events occuring
    int nL         = 0;     // New Latent clones being produced without mutation
    int nmL        = 0;     // New sequence starting with one latent clone
    int nmA        = 0;     // New sequence starting with one active cell
    int nD         = 0;     // New defective sequence//clone
    float nReact   = 0;     //Number of reactivations during that time step
    
    for (int i = 0; i < nSequences; i++){
        Ltotal   += SEQUENCES.at(2).second.at(i);
        Atotal   += SEQUENCES.at(3).second.at(i);
        Vtotal   += SEQUENCES.at(4).second.at(i);
        LXRtotal += SEQUENCES.at(5).second.at(i);
        nLclones += SEQUENCES.at(7).second.at(i);
        nL       += SEQUENCES.at(8).second.at(i); //new latent clones not mut
        nmL      += SEQUENCES.at(9).second.at(i); //new latent clones mutated
        nmA      += SEQUENCES.at(10).second.at(i);//new active clones mutated
        nD       += SEQUENCES.at(11).second.at(i);//new defective clones
        nEvents  += SEQUENCES.at(12).second.at(i);
        nReact   += SEQUENCES.at(13).second.at(i);
        if (SEQUENCES.at(0).second.at(i) > idmut){
        	Lmut     += SEQUENCES.at(2).second.at(i);
        	Amut     += SEQUENCES.at(3).second.at(i);
        	Vmut     += SEQUENCES.at(4).second.at(i);
        	nMclones += SEQUENCES.at(7).second.at(i);
        }
    }
    
    for (int i = 0; i < nDclones; i++){
    	Dtotal += DEFECTIVES.at(2).second.at(i);
    }
    
    
    // // Vectors of vectors for clone-specific information // // 
    // First index is the sequence, second the clone.
    std::vector<std::vector<int> > ID_CL;
    std::vector<std::vector<float> > REACTIVATION;
    std::vector<std::vector<float> > LATENT;
    std::vector<std::vector<float> > ANTIGEN;
    // Temporal vectors needed to full the vectors above
    std::vector<int> ID_CLtemp;
    std::vector<float> Rtemp;
    std::vector<float> Ltemp;
    std::vector<float> AGtemp;
    
    std::vector<int> ID3counter; //length equal to length of nSequences
    std::vector<float> dtimevar; //auxiliary for hybrid simulation
    std::vector<int> dtimeaction; //auxiliary for hybrid simulation
    std::vector<int> dLatentAction; //auxiliary for hybrid simulation
    
    // Accumulated total number of clones already read from the input files
    int accumulated = 0;
    // Filling the vectors of vectors from the input files
    for(int i=0; i<nSequences; i++){
    	for(int j=0; j<SEQUENCES.at(7).second.at(i); j++){
    		ID_CLtemp.push_back(CLONES.at(0).second.at(accumulated+j));
    		Rtemp.push_back(CLONES.at(1).second.at(accumulated+j));
    		Ltemp.push_back(CLONES.at(2).second.at(accumulated+j));
    		AGtemp.push_back(CLONES.at(3).second.at(accumulated+j));
    	}
    	ID_CL.push_back(ID_CLtemp);
    	REACTIVATION.push_back(Rtemp);
    	LATENT.push_back(Ltemp);
    	ANTIGEN.push_back(AGtemp);
    	ID_CLtemp.clear();
    	Rtemp.clear();
    	Ltemp.clear();
    	AGtemp.clear();
    	accumulated += SEQUENCES.at(7).second.at(i);
    	
    	//array of counters for IDclone
    	if (SEQUENCES.at(7).second.at(i) == 0){
    		ID3counter.push_back(0); //if no clones exist the counter starts at 0
    	}
    	else{
    		//if clones exist ID starts at the latest IDclone for that sequence +1
    		ID3counter.push_back(ID_CL.at(i).at(SEQUENCES.at(7).second.at(i)-1)+1);
    	}
        dtimevar.push_back(dt);
        dtimeaction.push_back(0);
        dLatentAction.push_back(0);
    }
    CLONES.clear(); //No longer useful
    
    // // Creating output files // //
    //Totals output file
    FILE *outTOT = fopen("output/totals.csv","w");
    fprintf(outTOT,"t,T,nLclones,nMclones,nDclones,nSequences,Ltotal,Atotal,Vtotal,LXRtotal,Lmut,Amut,Vmut,Dtotal,nL,nmL,nmA,nD,nEvents,nReact\n");
    fprintf(outTOT, "%f,%f,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d,%d,%d,%f\n", tStart, T, nLclones, nMclones, nDclones, nSequences, Ltotal, Atotal, Vtotal, LXRtotal, Lmut, Amut, Vmut, Dtotal, nL, nmL, nmA, nD,nEvents,nReact);
    fclose(outTOT);
    //sequences and clones output files
    FILE *outSEQ = fopen("output/sequences.csv","w");
    FILE *outCLO = fopen("output/clones.csv","w");
    fprintf(outSEQ, "t,ID_created,ID_sequence,L,A,V,LXR,Nmut,Nclones,nLs,nmLs,nmAs,nDs,nEventss,nReacts\n");
	fprintf(outCLO, "t,ID_created,ID_sequence,ID_clone,r,L,Ag\n");
    for(int i=0; i < nSequences; i++){
        fprintf(outSEQ,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", tStart, SEQUENCES.at(0).second.at(i), SEQUENCES.at(1).second.at(i), SEQUENCES.at(2).second.at(i), SEQUENCES.at(3).second.at(i), SEQUENCES.at(4).second.at(i), SEQUENCES.at(5).second.at(i), SEQUENCES.at(6).second.at(i), SEQUENCES.at(7).second.at(i), SEQUENCES.at(8).second.at(i), SEQUENCES.at(9).second.at(i), SEQUENCES.at(10).second.at(i), SEQUENCES.at(11).second.at(i), SEQUENCES.at(12).second.at(i), SEQUENCES.at(13).second.at(i));
        for(int j=0; j < SEQUENCES.at(7).second.at(i); j++){
        	fprintf(outCLO, "%f,%f,%f,%d,%f,%f,%f\n", tStart, SEQUENCES.at(0).second.at(i), SEQUENCES.at(1).second.at(i), ID_CL[i][j], REACTIVATION[i][j], LATENT[i][j], ANTIGEN[i][j]);
        }
    }
    fclose(outSEQ);
    fclose(outCLO);
    // defectives output file
    FILE *outDEF = fopen("output/defectives.csv","w");
    fprintf(outDEF, "t,ID_created,ID_sequence,D,Ag\n");
    for(int i=0; i < nDclones; i++){
    	fprintf(outDEF, "%f,%f,%f,%f,%f\n", tStart, DEFECTIVES.at(0).second.at(i), DEFECTIVES.at(1).second.at(i), DEFECTIVES.at(2).second.at(i), DEFECTIVES.at(3).second.at(i));
    }
    fclose(outDEF);
    
    int ID_created  = round(tStart*100);
    int ID_sequence  = 0;
    int ID_sequenceD = 0;
    int count  = 0;
//    int count2 = 0;
    int time = (tEnd - tStart)/dt;
    float t = 0;
    
//    std::default_random_engine generator;
//    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
//    std::normal_distribution<double> distribution(0.0,1.0);

    
    float Vtotal2 = 0;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    int nthre = 12;
    
    // Antigen parameters
    float lam = 60.00;
    float sA  = 30*1.96*1.0e+7;
    float pInteract = 1.0e-7;
    float a0 = 1.0;
    float Ksq = 1.0;
    float Asq = sA*pInteract*(pow(a0,2.0))*Ksq/lam;

    float idmut = monthsART*100;
    
    //generators
                
    std::default_random_engine generator;
    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::normal_distribution<double> distribution(0.0,1.0);

    std::normal_distribution<double> distribution_PR(-1.0,0.8); // pR distribution normal
    std::normal_distribution<double> distribution3(0,sqrt(Asq)); //a0 initialization
    
    
    for (int i=0; i<time; i++){
        t = i*dt+dt+tStart;
        ID_created++;
        ID_sequence  = 0;
        ID_sequenceD = 0;
        if (nSequences>0){
            count++;
//            count2++;
            // beta on steady state
            if ((t > (1.0+0.009)) && (t < (1.0+0.011))){
//                beta = 17.59*(pow(10,-order-3));
//                beta = 18.80*(pow(10,-order-3)); // this is the good one for exponential
                // 19 was working good
                
                beta =18*(pow(10,-order-3));
                
                
                
                
                //16.35 with latent
                //16.4 too big at order 6 and pL 0.1 and 16.37 too low
                //16.1 works at order 5 and pL 0.001
                
            }

            // beta on ART
            if ((t > (monthsART+0.009)) && (t < (monthsART+0.011))){
//                beta = 2.9*(pow(10,-order-3)); this is the good one for exponential
//                beta = 0.5*(pow(10,-order-4));
                beta = 6*(pow(10,-order-3));
                
                
//                6
//                5

//                0.01
                
//                1.0 was
            }
            
            
            int nEventsTotal = round(beta*T*Vtotal*dt);
            
//            std::cout <<nEventsTotal << " events \n";
            
            std::discrete_distribution<> d_event(SEQUENCES.at(4).second.begin(), SEQUENCES.at(4).second.end());
            
//            for (int ii=0; ii<5; )
            
            std::vector<int> events_seq;
            for (int ii=0; ii<SEQUENCES.at(4).second.size(); ii++){
                events_seq.push_back(0);
            }
//
//            int events_seq[SEQUENCES.at(4).second.size()];
//            memset( events_seq, 0, SEQUENCES.at(4).second.size()*sizeof(int) );

            for (int ii=0; ii<nEventsTotal; ii++){
                int number = d_event(generator);
                events_seq.at(number) += 1;
//                ++events_seq[number];
            }
            
            Vtotal2 = Vtotal; //I use Vtotal2 for T cell dynamics which has to use Vtotal as of previous step not as of the new step
            //Resetting to zero total variables
            Ltotal   = 0;
            Atotal   = 0;
            Vtotal   = 0;
            LXRtotal = 0;
            Lmut     = 0;
            Amut     = 0;
            Vmut     = 0;
            Dtotal   = 0;
            nEvents  = 0;
            nReact   = 0;
            nL       = 0;
            nmL      = 0;
            nmA      = 0;
            nD       = 0;
            nLclones = 0;
            nMclones = 0;
                   
            // Before the next for loop use this line if running in cluster:
            // #pragma omp parallel for num_threads(128)
            // or these 2 lines if running in my macbook
            // omp_set_dynamic(0);
            // #pragma omp parallel for	
            
            
            // Loop over the existing sequences (this is the one being paralellized
//            omp_set_dynamic(0);
//            omp_set_num_threads(nthre);
            #pragma omp parallel for num_threads(nthre)
            for (int j = 0; j<nSequences; j++){
                evolve_sequences(SEQUENCES.at(2).second.at(j), SEQUENCES.at(3).second.at(j), SEQUENCES.at(4).second.at(j), SEQUENCES.at(5).second.at(j), SEQUENCES.at(7).second.at(j), SEQUENCES.at(8).second.at(j), SEQUENCES.at(9).second.at(j), SEQUENCES.at(10).second.at(j), SEQUENCES.at(11).second.at(j), SEQUENCES.at(12).second.at(j), SEQUENCES.at(13).second.at(j), dtimevar.at(j), dtimeaction.at(j), dLatentAction.at(j), beta, T, REACTIVATION.at(j), LATENT.at(j), ANTIGEN.at(j), ID_CL.at(j), ID3counter.at(j), omp_get_thread_num(), events_seq[j]);
            }
            
            

            
            
            //Dynamics for defectives
            #pragma omp parallel for num_threads(nthre)
            for (int j=0; j<nDclones; j++){
                evolve_defectives(DEFECTIVES.at(2).second.at(j), DEFECTIVES.at(3).second.at(j), omp_get_thread_num());
            }
            
            // // Calculating totals // //
            for (int j=0; j<nSequences; j++){
                // Updating totals of new clones and sequences being created as well as events happening
                LXRtotal += SEQUENCES.at(5).second.at(j);
                nL       += SEQUENCES.at(8).second.at(j);
                nmL      += SEQUENCES.at(9).second.at(j);
                nmA      += SEQUENCES.at(10).second.at(j);
                nD       += SEQUENCES.at(11).second.at(j);
                nEvents  += SEQUENCES.at(12).second.at(j);
                nReact   += SEQUENCES.at(13).second.at(j);
                
                //Updating totals
                nLclones += LATENT.at(j).size()+SEQUENCES.at(9).second.at(j); //update number of latent clones
                Ltotal   += SEQUENCES.at(2).second.at(j)+SEQUENCES.at(9).second.at(j);
                Atotal   += SEQUENCES.at(3).second.at(j)+SEQUENCES.at(10).second.at(j);
                Vtotal   += SEQUENCES.at(4).second.at(j);
                Dtotal   += SEQUENCES.at(11).second.at(j);
                if (SEQUENCES.at(0).second.at(j) > idmut){
                    nMclones += LATENT.at(j).size();
                    Lmut += SEQUENCES.at(2).second.at(j);
                    Amut += SEQUENCES.at(3).second.at(j);
                    Vmut += SEQUENCES.at(4).second.at(j);
                }
                if (round(t*100)>idmut){
                    nMclones += SEQUENCES.at(9).second.at(j);
                    Lmut += SEQUENCES.at(9).second.at(j);
                    Amut += SEQUENCES.at(10).second.at(j);
                }
            }
            
            
            // Antigen parameters
            float lam = 60.00;
            float sA  = 30*1.96*1.0e+7;
            float pInteract = 1.0e-7;
            float a0 = 1.0;
            float Ksq = 1.0;
            float Asq = sA*pInteract*(pow(a0,2.0))*Ksq/lam;

            float idmut = monthsART*100;
            
            //generators
                        
            std::default_random_engine generator;
            generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::normal_distribution<double> distribution(0.0,1.0);

            std::normal_distribution<double> distribution_PR(-1.0,0.8); // pR distribution normal
        //                std::normal_distribution<double> distribution_PR(-1,1);
            std::normal_distribution<double> distribution3(0,sqrt(Asq)); //a0 initialization
            
            
            //Create new sequences
            for (int j=0; j<nSequences; j++){
                
                
            
            	//Create mutants with L=1, A=0
                float lpr = 0;
                float a00 = 0;
            	for (int k=0; k<SEQUENCES.at(9).second.at(j)*pow(10,(orderL-order)); k++){
            		SEQUENCES.at(0).second.push_back(ID_created);
            		SEQUENCES.at(1).second.push_back(ID_sequence);
            		ID_sequence++;
            		ID_CLtemp.push_back(0);
            		
//            		do{
//            			lpr = -distribution2(generator);
//            		}while(lpr<-3.2);
                    
                    do{
                        lpr = distribution_PR(generator);
                    }while(lpr<-4.0 or lpr>0);
                    
            		Rtemp.push_back(pow(10,lpr));
            		Ltemp.push_back(1);
            		a00 = distribution3(generator);
            		if (a00<0){
            		a00 = 0;
            		}
            		AGtemp.push_back(a00);
            		ID_CL.push_back(ID_CLtemp);
            		REACTIVATION.push_back(Rtemp);
            		LATENT.push_back(Ltemp);
            		ANTIGEN.push_back(AGtemp);
                    ID3counter.push_back(0);
                    dtimevar.push_back(dt);
                    dtimeaction.push_back(0);
                    dLatentAction.push_back(0);
            		ID_CLtemp.clear();
            		Rtemp.clear();
            		Ltemp.clear();
            		AGtemp.clear();
            		SEQUENCES.at(2).second.push_back(1);
            		SEQUENCES.at(3).second.push_back(0);
            		SEQUENCES.at(4).second.push_back(0);
            		SEQUENCES.at(5).second.push_back(pow(10,lpr));
            		if (idmut < ID_created){
            		SEQUENCES.at(6).second.push_back(SEQUENCES.at(6).second.at(j)+1);
            		}
            		else{
            		SEQUENCES.at(6).second.push_back(SEQUENCES.at(6).second.at(j));
            		}
            		SEQUENCES.at(7).second.push_back(1);
            		SEQUENCES.at(8).second.push_back(0);
            		SEQUENCES.at(9).second.push_back(0);
            		SEQUENCES.at(10).second.push_back(0);
            		SEQUENCES.at(11).second.push_back(0);
            		SEQUENCES.at(12).second.push_back(0);
                    SEQUENCES.at(13).second.push_back(0);
            	}
            	
            	
            	//Create mutants with L=0, A=1
            	for (int k=0; k<SEQUENCES.at(10).second.at(j); k++){
            		SEQUENCES.at(0).second.push_back(ID_created);
            		SEQUENCES.at(1).second.push_back(ID_sequence);
            		ID_sequence++;
            		ID_CL.push_back(ID_CLtemp);
            		REACTIVATION.push_back(Rtemp);
            		LATENT.push_back(Ltemp);
            		ANTIGEN.push_back(AGtemp);
                    ID3counter.push_back(0);
                    dtimevar.push_back(dt);
                    dtimeaction.push_back(0);
                    dLatentAction.push_back(0);
            		SEQUENCES.at(2).second.push_back(0);
            		SEQUENCES.at(3).second.push_back(1);
            		SEQUENCES.at(4).second.push_back(0);
            		SEQUENCES.at(5).second.push_back(0);
            		if (idmut < ID_created){
            		SEQUENCES.at(6).second.push_back(SEQUENCES.at(6).second.at(j)+1);
            		}
            		else{
            		SEQUENCES.at(6).second.push_back(SEQUENCES.at(6).second.at(j));
            		}
            		SEQUENCES.at(7).second.push_back(0);
            		SEQUENCES.at(8).second.push_back(0);
            		SEQUENCES.at(9).second.push_back(0);
            		SEQUENCES.at(10).second.push_back(0);
            		SEQUENCES.at(11).second.push_back(0);
            		SEQUENCES.at(12).second.push_back(0);
                    SEQUENCES.at(13).second.push_back(0);
            	}
            	//Create defectives
            	for (int k=0; k<SEQUENCES.at(11).second.at(j); k++){
            		DEFECTIVES.at(0).second.push_back(ID_created);
                    DEFECTIVES.at(1).second.push_back(ID_sequenceD);
                    ID_sequenceD++;
                    DEFECTIVES.at(2).second.push_back(1);
                    a00 = distribution3(generator);
                    if (a00<0){
                    a00 = 0;
                    }
                    DEFECTIVES.at(3).second.push_back(a00);
            	}
            }
            
            //Erase empty sequences and defectives
//            omp_set_dynamic(0);
            #pragma omp parallel for num_threads(2)
            for (int l=0; l<2; l++){
                if(l==0){
                    //Erase empty sequences
                    for (int j=nSequences-1; j>=0; j= j-1){
                        if (SEQUENCES.at(2).second.at(j)==0 && SEQUENCES.at(3).second.at(j)==0 && SEQUENCES.at(4).second.at(j)==0){
                            SEQUENCES.at(0).second.erase(SEQUENCES.at(0).second.begin()+j);
                            SEQUENCES.at(1).second.erase(SEQUENCES.at(1).second.begin()+j);
                            SEQUENCES.at(2).second.erase(SEQUENCES.at(2).second.begin()+j);
                            SEQUENCES.at(3).second.erase(SEQUENCES.at(3).second.begin()+j);
                            SEQUENCES.at(4).second.erase(SEQUENCES.at(4).second.begin()+j);
                            SEQUENCES.at(5).second.erase(SEQUENCES.at(5).second.begin()+j);
                            SEQUENCES.at(6).second.erase(SEQUENCES.at(6).second.begin()+j);
                            SEQUENCES.at(7).second.erase(SEQUENCES.at(7).second.begin()+j);
                            SEQUENCES.at(8).second.erase(SEQUENCES.at(8).second.begin()+j);
                            SEQUENCES.at(9).second.erase(SEQUENCES.at(9).second.begin()+j);
                            SEQUENCES.at(10).second.erase(SEQUENCES.at(10).second.begin()+j);
                            SEQUENCES.at(11).second.erase(SEQUENCES.at(11).second.begin()+j);
                            SEQUENCES.at(12).second.erase(SEQUENCES.at(12).second.begin()+j);
                            SEQUENCES.at(13).second.erase(SEQUENCES.at(13).second.begin()+j);
                            ID_CL.erase(ID_CL.begin()+j);
                            REACTIVATION.erase(REACTIVATION.begin()+j);
                            LATENT.erase(LATENT.begin()+j);
                            ANTIGEN.erase(ANTIGEN.begin()+j);
                            ID3counter.erase(ID3counter.begin()+j);
                            dtimevar.erase(dtimevar.begin()+j);
                            dtimeaction.erase(dtimeaction.begin()+j);
                            dLatentAction.erase(dLatentAction.begin()+j);
                        }
                    }
                    
                }
                if(l==1){
                    // Erase empty defectives
                    for (int j=nDclones-1; j>=0; j= j-1){
                        Dtotal += DEFECTIVES.at(2).second.at(j);
                        if (DEFECTIVES.at(2).second.at(j)==0){
                            DEFECTIVES.at(0).second.erase(DEFECTIVES.at(0).second.begin()+j);
                            DEFECTIVES.at(1).second.erase(DEFECTIVES.at(1).second.begin()+j);
                            DEFECTIVES.at(2).second.erase(DEFECTIVES.at(2).second.begin()+j);
                            DEFECTIVES.at(3).second.erase(DEFECTIVES.at(3).second.begin()+j);
                        }
                    }
                
                }
            }
            
            nSequences = SEQUENCES.at(0).second.size();
            nDclones = DEFECTIVES.at(0).second.size();
            
            
            
            // Dynamics of T cells
            T+= (-(beta*Vtotal2+mu_T)*T + lam_T)*dt + sqrt((lam_T + T*(beta*Vtotal2+mu_T))*dt)*distribution(generator);
            

            FILE *outTOT = fopen("output/totals.csv","a");
                fprintf(outTOT, "%f,%f,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d,%d,%d,%f\n", t, T, nLclones, nMclones, nDclones, nSequences, Ltotal, Atotal, Vtotal, LXRtotal, Lmut, Amut, Vmut, Dtotal, nL, nmL, nmA, nD,nEvents, nReact);
            fclose(outTOT);
            
            if (count == 100){
//               omp_set_dynamic(0);
               #pragma omp parallel for num_threads(2)
               for (int l=0; l<2; l++){
                   if(l==0){
                       FILE *outSEQ = fopen("output/sequences.csv","a");
                       FILE *outCLO = fopen("output/clones.csv","a");
                       for(int j=0; j < nSequences; j++){

                           fprintf(outSEQ,"%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", (int)round(t), SEQUENCES.at(0).second.at(j), SEQUENCES.at(1).second.at(j), SEQUENCES.at(2).second.at(j), SEQUENCES.at(3).second.at(j), SEQUENCES.at(4).second.at(j), SEQUENCES.at(5).second.at(j), SEQUENCES.at(6).second.at(j), SEQUENCES.at(7).second.at(j), SEQUENCES.at(8).second.at(j), SEQUENCES.at(9).second.at(j), SEQUENCES.at(10).second.at(j), SEQUENCES.at(11).second.at(j), SEQUENCES.at(12).second.at(j), SEQUENCES.at(13).second.at(j));


                           for(int k=0; k < SEQUENCES.at(7).second.at(j); k++){
                               fprintf(outCLO, "%d,%f,%f,%d,%f,%f,%f\n", (int)round(t), SEQUENCES.at(0).second.at(j), SEQUENCES.at(1).second.at(j), ID_CL[j][k], REACTIVATION[j][k], LATENT[j][k], ANTIGEN[j][k]);
                           }
                       }
                       fclose(outSEQ);
                       fclose(outCLO);
                   }
                   if(l==1){
                       FILE *outDEF = fopen("output/defectives.csv","a");
                       for(int j=0; j < nDclones; j++){
                           fprintf(outDEF, "%d,%f,%f,%f,%f\n", (int)round(t), DEFECTIVES.at(0).second.at(j), DEFECTIVES.at(1).second.at(j), DEFECTIVES.at(2).second.at(j), DEFECTIVES.at(3).second.at(j));
                       }
                       fclose(outDEF);
                   }
               }

               count = 0;
           }
            
            std::cout << t << " time \n";
        }   // Close of if nC >0
    }       // Close of for loop
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t1-t0);
    std::cout << nthre  << " threads " << duration.count()/60 << " minutes \n";
    std::cout << '\a';
}           //close of main

// To test when there is no contribution from LR, do this:
// comment line 336 (A>200)
// A<200 comment lines:
// 446-450
// 635-639
