# Non-Thermal-distributions

This program computes the dark matter abundance of a toy model by tracking the kinematic distributions. 

The code is based on:
"Impact of non-thermal phase-space distributions on dark matter abundance in secluded sectors", Hugues Beauchesne and Cheng-Wei Chiang, 2401.03657 [hep-ph].

Required Python libraries:
- math
- numpy
- scipy
- matplotlib

How to execute:

  python3 main.py <NProcesses> <Result_folder> <Gamma_B>
  
where:
- NProcesses: Number of threads to use
- Result_folder: Name of the folder to save the results
- Gamma_B: Width of $\phi_B$ in GeV
  
Example:

    python3 main.py 8 R1 1e-17

Notes:
- Settings are stored in main.py.
- The code is numerically very demanding in terms of both CPU and memory.
- The code is written in python3 format. Issues might arise if using python2.
