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
#include "tools.h"  // Numerical tools
#include <chrono>
#include <omp.h>

//#include <Eigen/Dense>

// Latent cells parameters
float nu_L = 0.00+25;
float mu_L = 7.195+25; //0.75 approx works  6.35

// Antigen parameters
float lam = 60.00;
float sA  = 30*1.96*1.0e+7;
float pInteract = 1.0e-7;
float a0 = 1.0;
float Ksq = 1.00;
float Asq = sA*pInteract*(pow(a0,2.0))*Ksq/lam;



//// Desponds parameters
//
//float nu_L = 0.00+25; //division rate of latent cells
//    float mu_L = 7.215+25; //death rate of latent cells
//
//float lam = 30*2.00;
//float sA  = 30*1.96*1.0e+7;
//float pInteract = 1.0e-7;
//float a0 = 1.0;
//float Ksq = 2.00;
//float Asq = sA*pInteract*(pow(a0,2.0))*Ksq/lam;

//float newCL = 200*30/10;



float newCL = 900;

// time variables

float dt = 0.01;
float dtt = 0.0001;
//float tEnd = 250;
float tEnd = 400.00;


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


void write_csv(std::string filename, std::vector<std::pair<std::string, const std::vector<float> > > &dataset){
    // Make a CSV file with one or more columns of integer values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<int>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size
    
    // Create an output filestream object
    std::ofstream myFile(filename);
    // Send column names to the stream
    for(int j = 0; j < dataset.size(); ++j)
    {
        myFile << dataset.at(j).first;
        if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
    }
    myFile << "\n";
    // Send data to the stream
    for(int i = 0; i < dataset.at(0).second.size(); ++i)
    {
        for(int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).second.at(i);
            if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }
    // Close the file
    myFile.close();
}

void evolution(std::vector<std::pair<std::string, std::vector<float> > > &data, const int idx, float LT){
    float RR = data.at(2).second.at(idx);
    float LL = data.at(3).second.at(idx);
    float Aj = data.at(4).second.at(idx);
    float ngen = 0;
    
    
    // Latent cells parameters
    float nu_L = 0.00+25;
    float mu_L = 7.195+25; //0.75 approx works  6.35

    // Antigen parameters
    float lam = 60.00;
    float sA  = 30*1.96*1.0e+7;
    float pInteract = 1.0e-7;
    float a0 = 1.0;
    float Ksq = 1.00;
    float Asq = sA*pInteract*(pow(a0,2.0))*Ksq/lam;
    
    
    float dt = 0.01;
    float dtt = 0.0001;
    //float tEnd = 250;
    float tEnd = 400.00;
    
    
    std::default_random_engine generator;
    
//    RR = 0;  // This to analysis for microsoft, erase after
   
    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count() + omp_get_thread_num()*1000);
    std::normal_distribution<double> distribution(0.0,1.0);
   
        //Evolution of the clone
       LL  += (((1-2*RR)*(nu_L + Aj) - mu_L)*LL)*dt + sqrt((nu_L+Aj+mu_L)*LL*dt)*distribution(generator);
       
       
       // trying something new where only reactivates if antigen stimulated
//       LL  += (((1-2*RR)*Aj + nu_L- mu_L)*LL)*dt + sqrt((nu_L+Aj+mu_L)*LL*dt)*distribution(generator);
//       - (LL/LT)*pow(LT/(4e5),3)*dt
       
    
       
       
    // Dynamics of antigens
    for(int i=0; i<(dt/dtt); i++){
        ngen = sqrt(2. *lam *Asq *dtt) * distribution(generator);
        if (ngen<0){
            ngen=0;
        }
        
        Aj  += (-lam * Aj * dtt) + ngen;   // Dynamics of antigens
        if (Aj<0){
            Aj=0;
        }
    }
    if (LL<0.5){
        LL=0;
    }
    data.at(3).second.at(idx) = LL;
    data.at(4).second.at(idx) = Aj;
} // End of evolution void


int main(int argc, char **argv) {
    std::vector<std::pair<std::string, std::vector<float> > > TOT = read_csv("initial/clones_0.0.csv"); //Read the initial condition
    float tStart = TOT.at(0).second.at(0);  //Read the time where the simulation starts
    TOT.erase(TOT.begin()+0);               // Erase the vector of time
    int nC = TOT.at(0).second.size();       //Calculate number of clones
    
    FILE *out = fopen("output/clones.csv","w");
    fprintf(out,"t,ID1,ID2,r,L,aj\n");
    for(int i=0; i < nC; i++){
        fprintf(out,"%f,%f,%f,%f,%f,%f\n", tStart, TOT.at(0).second.at(i), TOT.at(1).second.at(i), TOT.at(2).second.at(i), TOT.at(3).second.at(i), TOT.at(4).second.at(i));
    }
    fclose(out);
    
    float L = 0;     // Total number of latent cells
    for (int i = 0; i < nC; i++){
        L += TOT.at(3).second.at(i);
    }
    
    float ID1 = TOT.at(0).second.at(nC-1);
    float ID2 = 0;
       
    FILE *out2 = fopen("output/totals.csv","w");
    fprintf(out2,"t,nClones,Ltotal\n");
    fprintf(out2, "%f,%d,%f\n", tStart, nC, L);
    fclose(out2);
    
   
    int count  = 0;
//    int count2 = 0;
    int time = (tEnd - tStart)/dt;
    float t = 0;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    
    
    std::normal_distribution<double> distribution2(-1.0, 0.8);
    
//    std::exponential_distribution<double> distribution2(0.5); // distribution of log pr for new clones
    
//    std::uniform_real_distribution<double> distribution2(0,5); // distribution of log pr for new clones
    
    std::normal_distribution<double> distribution3(0,sqrt(Asq));  //for initialitazion of a0
//    std::normal_distribution<double> distribution3(1,1);  //for initialitazion of a0

    
    for (int i=0; i<time; i++){
        t = i*dt+dt+tStart;
        ID1++;
        ID2 = 0;
       
        count++;
//            count2++;
        
        //Write here code for the evolution of the clones
//            #pragma omp parallel for num_threads(128)
        omp_set_dynamic(0);
        #pragma omp parallel for
        for (int j = 0; j<nC; j++){
            evolution(TOT, j, L);
        }
        
        float lpr = 0;
        float a00 = 0;
        
        if ((t > (60.0+0.009)) && (t < (60.0+0.011))){
            newCL = 1;
        }
        
        //Creating new clones
        
        for (int k=0; k < newCL; k++){
                // Creation of new clones that have 1 latent cell
                TOT.at(0).second.push_back(ID1);                 // Create value of ID1
                TOT.at(1).second.push_back(ID2);                 // Create value of ID2
                ID2++;                                          //ID2 value increases by one
                                                         
//                do{
//                    lpr = -distribution2(generator);
////                    lpr = -1;
//                }while( lpr<-5 || lpr>0 );      //
                
                do{
                    lpr = distribution2(generator);
                }while(lpr<-4.0 or lpr>0);
                
                TOT.at(2).second.push_back(pow(10,lpr)); // Create pR
                TOT.at(3).second.push_back(1.0);
                a00 = distribution3(generator);
                if (a00<0){
                a00 = 0;
                }
                TOT.at(4).second.push_back(a00);                      // Create value of aj

//                TOT.at(2).second.push_back(0);                      // Create pR
//                TOT.at(3).second.push_back(1.0);                      // Create value of L
//                TOT.at(4).second.push_back(1.0);                    // create value of aj
            }
        
        
        for (int j=nC-1; j>=0; j= j-1){
            if (TOT.at(3).second.at(j) == 0){
                TOT.at(0).second.erase(TOT.at(0).second.begin()+j);
                TOT.at(1).second.erase(TOT.at(1).second.begin()+j);
                TOT.at(2).second.erase(TOT.at(2).second.begin()+j);
                TOT.at(3).second.erase(TOT.at(3).second.begin()+j);
                TOT.at(4).second.erase(TOT.at(4).second.begin()+j);
            }
        } //Finish erasing empty clones
        
        
        nC = TOT.at(0).second.size();
        L = 0;     // Total number of latent cells
        for (int i = 0; i < nC; i++){
            L += TOT.at(3).second.at(i);
            }
        
        
//            if (count2 == 10){
//                FILE *out2 = fopen("totals.csv","a");
//                fprintf(out2, "%f,%f,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", t, T, nC, L, A, V, M, LatND, nND, Lmut, Amut, Vmut, nmut);
//                fclose(out2);
//                count2 = 0;
//            }

        FILE *out2 = fopen("output/totals.csv","a");
        fprintf(out2, "%f,%d,%f\n", t, nC, L);
        fclose(out2);
        


//            int ti = static_cast<int>(t);
        
        if (count == 100){
            FILE *out = fopen("output/clones.csv","a");
            for(int i=0; i < nC; i++){
                fprintf(out,"%d,%f,%f,%f,%f,%f\n", (int)round(t), TOT.at(0).second.at(i), TOT.at(1).second.at(i), TOT.at(2).second.at(i), TOT.at(3).second.at(i), TOT.at(4).second.at(i));
            }
            fclose(out);
            count = 0;
        }
            std::cout << t << "\n";
    }       // Close of for loop
}           //close of main
