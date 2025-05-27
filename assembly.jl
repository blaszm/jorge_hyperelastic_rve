function assemble_global!(K, g, dh, cv, F_macro, mp_i, mp_m, u, states)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    # start_assemble resets K and g
    assembler = start_assemble(K, g)

    # Loop over all cells in the grid
    for (cell,state) in zip(CellIterator(dh),states)
        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs

        # choose material parameters depending on cell set 
        mp = cellid(cell) in getcellset(dh.grid, "inclusions") ? mp_i : mp_m

        assemble_element!(ke, ge, cell, cv, F_macro, mp, ue, state)
        assemble!(assembler, global_dofs, ke, ge)
    end
    return
end;

function assemble_element!(ke, ge, cell, cv, F_macro, mp, ue, state)
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)

    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u + F_macro # add macroscopic deformation gradient
        C = tdot(F) # F' ⋅ F
        # Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, mp)
        P = F ⋅ S
        I = one(S)
        ∂P∂F = otimesu(I, S) + 2 * F ⋅ ∂S∂C ⊡ otimesu(F', I)

        # Save deformation gradient, stress and energy in struct
        state[qp].F = F
        state[qp].P = P
        state[qp].Ψ = P ⊡ F

        # Loop over test functions
        for i in 1:ndofs
            # Test function gradient
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            ge[i] += (∇δui ⊡ P) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end

    return
end;