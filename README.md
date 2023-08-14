
# Overview

This repository contains and data and scripts for reproducing the results accompanying the manuscript  

### Clonal heterogeneity and antigenic stimulation shape persistence of the latent reservoir of HIV
Marco Garcia Noceda<sup>1</sup>, and John P. Barton<sup>1,2,3,#</sup>

<sup>1</sup> Department of Physics and Astronomy, University of California, Riverside  
<sup>2</sup> Department of Physics and Astronomy, University of Pittsburgh  
<sup>3</sup> Department of Computational and Systems Biology, University of Pittsburgh School of Medicine  
<sup>#</sup> correspondence to [jpbarton@pitt.edu](mailto:jpbarton@pitt.edu)  

The preprint is available at __INSERT LINK HERE__.

# Contents

Scripts for generating the distinct simulations performed can be found in `src`. These include `fullsimulation.cpp` as the full simulation of the model, `LatentOnly.cpp` as the constant production during ART approximation, `simulation_mutations.cpp` as the simulation used for estimating an upper limit in the number of potential accumulated mutations during ART and finally `EarlyART.cpp` is for the simulation of infection with early intervention either because of early ART or for elite controllers.

The `figures.ipynb` contains a template Jupyter notebook for reproducing the figures accompanying the paper. It makes use of the libraries `figures.py` and `mplot.py`.

Due to the large size of the output files generated by the simulations, data has been stored in a compressed format using Zenodo. To access the full set of data, navigate to the [Zenodo record](https://zenodo.org/record/7898811). Download and extract the contents of `data.zip` as the folder `/data`.



### Software dependencies

Simulations make use of C++11 and OpenMP to run in parallel.

### Compilation examples

To compile and run on mac:
`clang++ -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp fullsimulation.cpp -o fullsimulation.exe -std=c++11`
`./simulation.exe`

To compile and run on linux:
`g++ -fopenmp -std=c++11 fullsimulation.cpp -o fullsimulation.exe`
`./simulation.exe`

# License

This repository is dual licensed as [GPL-3.0](LICENSE-GPL) (source code) and [CC0 1.0](LICENSE-CC0) (figures, documentation, and our presentation of the data).
