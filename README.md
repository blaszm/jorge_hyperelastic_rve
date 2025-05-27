# jorge_hyperelastic_rve

Program for microscale calculations of periodic RVE with hyperelastic material model.

Usage: Run main.jl to start the program. First, define material model and parameters. Use the calculate_tangent function to obtain the macroscopic tangent (effective stiffness). For arbitrary loads, define deformation gradient F and run the solve function, with returns the average 1st Piola-Kirchhoff stress P and the average energy Psi = Int_Omega P : F dV. Options: Use "steps" to decide in how many equal parts the load should be split for calculation, where higher numbers take more time but should improve convergence (default: steps=10). Use "output" (=true) to obtain convergence information in the console while running the program and a resulting vtu.file which allows to look at the simulation results in ParaView.
