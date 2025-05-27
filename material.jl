# Note: you can very easily change / extend the material energy function here in order to enforce a different material model
struct NeoHooke
    μ::Float64
    λ::Float64
end

function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    IC_1 = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (IC_1 - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end

# Struct to store converged values of deformation gradient, 1st Piola Kirchhoff stress and energy
mutable struct MaterialState{T, S <: Tensor{2, dim, T} where dim}
    F::S
    P::S
    Ψ::T
end

# Constructor for struct MaterialState
function MaterialState()
    return MaterialState(zero(Tensor{2,2}),zero(Tensor{2,2}),0.0)
end

# Calculation of derivatives
function constitutive_driver(C, mp)
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;