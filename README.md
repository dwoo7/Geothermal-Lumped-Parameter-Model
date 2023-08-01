# Resource Consent Project - Rotorua Geyser Recovery - ENGSCI 263

The file ' main.py' calculates pressure and temperature values using ODEs modelled on the geyser recovery in Rotorua.
The file 'plots.py' produces plots of the given data for analysis.

***To locate files: Go into the data folder and select "main.py"***

***path ---> data/main.py***

Model: Plots the calculated numerical values against the observed data provided with the best fit line. (Generating these plots takes some time)

Benchmark: For both pressure and temperature, containing: benchmark, error analysis and time step convergence.

Unit tests: Tests the functationality of the implemented Improved Euler's Method

Calibration: Parameters are calibrated automatically using the curve_fit function from the scipy library.

Model use: Plots predictions for different production rates for both pressure and temperature within the region.

Uncertainty: Calculates 95% confidence intervals in the forecasts for the 5 different situations modelled.

To run following functions use "main.py" along with "gr_T.txt", "gr_p.txt", "gr_q1.txt", "gr_q2.txt" (text files containing monitoring data sets) in the same directory or working folder.

The ".png" graphs are outputs for the functions specified in the file names.

To plot specific functions of the code you can use  the boolean expressions in the main function in "main.py":

"plot_model = True" ---> plot both temperature and pressure with their best fit models

"plot_benchmarks = True" ---> plots the analytical vs numerical solution, error analysis and convergence tests for both pressure and temperature

"plot__future = True" ---> plots the forecasts of 5 different sitations on top of the preexisting best fit model for pressure and temperature

"plot_uncertainty = True" ---> plots the uncertainty or 95% confidence intervals for the forecasts of 5 unique situations.

***The following lines of code to operature the functions in "main.py" are found from line 709 - 712*** 