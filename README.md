# mhd-solver: Computational Physics Term Project
# A Multi-Dimensional, Distributed, Non-Oscillatory Ideal Magnetohydrodynamics Central Differencing Framework
## Created 03/21
###  Neil Baker, C. Michael Haynes, Corinne Hill, Eden Schapera, Kenneth Thompson

Please contact mhaynes@eas.gatech.edu for inquiries.
## Kenneth Actual Comments
I worked a bit around to be able to run multiple cases at once, but it seems to parallelize perfectly so not too worried.

Changes: 
- got rid of config file and moved everything to the initialize function so I could parallelize it.
- I set up the sigmoid case and all cases for it
- fully coded and parallelized the changing alpha cases and changing sigmoid cases in the 'run_me.py' file
  - for a difference set of alpha change the alpha_vals or for sigmoid change the sigmoid_vals
  - uncomment each one individually to run them because I was lazy and didn't want to be clever
  - the fixed_work_... sets all of the input values that are not being changed through the itteration
  - 
- the animate function now returns the vectors as well as plotting (and is not built for the alpha and sigmoid cases)
  - it saves the data instead of just plotting it now (needed to be done for what we want with the parrelization)
- I have a small thing to grab the data from the files after the run for plotting. Eden was posting stuff about plotting so I was waiting for that before I collected the data together
  - might take a few minutes of thought to extract the data exactly how we want. I just wanted to put something down that was general
- Started working on some of the other cases, but had some questions regarding what you guys were doing for initial conditions
  - specifically the en/10 stuff. honestly didn't think about it much, but wanted to make sure
- I added some thoughts as to how to do the adaptive time step. It seems pretty easy, but I probably underthought it
  - I would change the value of dt if the max velocity equals 0 OR implement the break condition if the CFL is 0 or nan
  - I did some goofy stuff with the eigenvalues of the 1-D MHD equations at the bottom of the MHD file using

$$
a^2 = \frac{\gamma p}{\rho} \hspace{5mm} B^2 = B_x^2 + B_y^2 + B_z^2\hspace{5mm} c_{ax} = \frac{B_x}{\sqrt{\rho}}\hspace{5mm} c_A^2 = \frac{B^2}{\rho}
$$
and plug that into 
$$
c_{f,s}^2 = \frac{1}{2} \left( a^2 + c_A^2 \pm \sqrt{(a^2 + c_A^2)^2 - 4a^2 c_{ax}^2} \right)
$$

other misc comments: 
- line 119 in Library does this divide by rho twice?
- I think you were passing in length=1 but had a domain from -1 to 1 so just make sure that is correct
- I changed the name from mhd-0 to mhd because it was being stupid when I tried to pass it in
- it might be worth while to stop the simulation after the CFL value goes to 0 or nan so conserve space and computing power
  - it goes much faster when it does so it doesn't really matter, but over many runs might be worth while
- 
