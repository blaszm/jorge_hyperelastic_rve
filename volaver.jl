# Calculate all necessary volume averages
function volaver(cv, dh, states, dim)

    # Number of quadrature points
    n_qp = getnquadpoints(cv)

    # Helper variable for total volume
    Ω = 0.0

    # Average 1st Piola-Kirchhoff stress
    P_aver = zero(Tensor{2, dim, Float64})

    # Average energy
    Ψ_aver = 0.0
    
    @inbounds for (no, cell) in enumerate(CellIterator(dh)) 
        reinit!(cv, cell)
        state = states[no]
        @inbounds for qp_i in 1:n_qp 
            dΩ =  getdetJdV(cv, qp_i)
            Ω += dΩ

            P_aver += state[qp_i].P * dΩ
            Ψ_aver += state[qp_i].Ψ	* dΩ
        end
    end

    P_aver /= Ω
    Ψ_aver /= Ω
    
    return P_aver, Ψ_aver
end