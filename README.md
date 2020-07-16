# seqfk
Mechanical DNA sequence analysis using monte carlo simulation of a generalised Frenkel Kantorova model


# files

* seqfk/
  * Parameterization
    contains the rigid base pair model data copied from [NucleosomeMMC](https://github.com/SchiesselLab/NucleosomeMMC)
  * analysis.py contains the analysis method
  * data.py contains classes to work with datasets and data storage
  * generate.py contains the sequence generation method that runs the monte carlo method and stores the resuls
  * model.py contains the energy model with the different potentials used
  * monte_carlo.py contains the constructor/compiler classes for the monte carlo methods
  * parameters.py reads in the rigid base pair parameters
  * sequence.py contains the sequence class used for storing the sequence and positions together and calling the monte carlo method
