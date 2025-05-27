# Code for simple 2D microscale RVE simulations of hyperelastic material
# M. Blaszczyk - TU Braunschweig - 2025

using Ferrite, FerriteGmsh, SparseArrays, LinearAlgebra, Tensors, ProgressMeter, IterativeSolvers

include("material.jl")
include("assembly.jl")
include("volaver.jl")
include("solver.jl")
include("writeoutput.jl")

#-----------------------------------------------------------

# Material parameters

# Inclusion
E = 20.0
ν = 0.3
μ = E / (2(1 + ν))
λ = (E * ν) / ((1 + ν) * (1 - 2ν))
mp_i = NeoHooke(μ, λ)

# Matrix
E = 10.0
ν = 0.3
μ = E / (2(1 + ν))
λ = (E * ν) / ((1 + ν) * (1 - 2ν))
mp_m = NeoHooke(μ, λ)

# Macroscopic tangent
#∂F∂P_macro = calculate_tangent(mp_i, mp_m, "periodic-rve.msh") 
#println(eigvals(∂F∂P_macro))

# Single solver
F_macro = SymmetricTensor{2, 2}([0.0 0.0; 0.0 0.01])
P_aver, Ψ_aver = solve(F_macro, mp_i, mp_m, "periodic-rve.msh"; output=true);