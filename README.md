#### mhd-solver: Computational Physics Term Project (PHYS 6260)                            
# A One-Dimensional, Distributed, Non-Oscillatory Ideal Magnetohydrodynamics Central Differ
## Created 03/21                                                                           
### C. Michael Haynes, Neil Baker, Kenneth Thompson, Corinne Hill, Eden Schapera
## Updated, Finalized & Submitted 04/29



# A brief code overview:

#### Each file carries out a particular routine in the production line between input conditions and finalized data
### The following files contain the entirety of what generated the ideal MHD results found in our report.


## mhd.py
#### This file represents the kernel of the solver. It includes the methods described in our model description and the work of Balbas et al. (2004).
#### Note: mhd_adaptive.py is a clone of mhd.py, with the recently developed adaptive timestepping routines implemented in a preliminary sense.

## Library.py
#### This file contains all of the helper functions defined to carry out specific routines.

## run_me.py
#### This file initializes all cases that we studied in this work. It stores initial configurations and gives the ability to instantiate a single instance or a distributed execution of multiple runs.

### datavis.py
#### This file contains a single function called for outputting & plotting in run_me and b

### b.py
#### This file processes the data that gets output at benchmarked intervals during runtime, converting it into plots. Animations are constructed directly from the CLI (using convert im6).

