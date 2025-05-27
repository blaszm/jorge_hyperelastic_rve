# Function to write output data, which can be viewn in Paraview
function collect_scalar_quantity(dh::DofHandler, states, symbol) 
    
    field_out = [Vec{1,Float64}[] for _ in 1:getncells(dh.grid)] # Array(Array(Tensor))
    field_i = [0.0] # one component array

    for (elmt, cell_states) in enumerate(states)
        field_cell = field_out[elmt]
        
        for state in cell_states
            field_i[1] = getfield(state,symbol)
            field_t = reinterpret(Vec{1, Float64}, vec(field_i)) # convert array to tensor
            push!(field_cell, field_t[1]) # add tensor to Array(Array()) 
        end
    end

    return field_out
end

function collect_tensor_quantity(dh::DofHandler, dim, states, symbol) 
    field_out = [Tensor{2, dim, Float64, dim^2}[] for _ in 1:getncells(dh.grid)]

    for (elmt, cell_states) in enumerate(states)
        field_cell = field_out[elmt]
        for state in cell_states
            field = getfield(state,symbol)
            push!(field_cell, field) 
        end
    end

    return field_out
end

function writeoutput(ip_h, qr, dh, u, states, dim)

    # Material visualization
    material_number = zeros(getncells(dh.grid))
    for cell in CellIterator(dh)
        material_number[cellid(cell)] = cellid(cell) in getcellset(dh.grid, "inclusions") ? 1 : 2
    end

    F_qp = collect_tensor_quantity(dh, dim, states, :F) # Vector of second order tensors
    P_qp = collect_tensor_quantity(dh, dim, states, :P)
    Ψ_qp = collect_scalar_quantity(dh, states, :Ψ)
    
    projector = L2Projector(ip_h, dh.grid)

    F_nodes = project(projector, F_qp, qr)
    P_nodes = project(projector, P_qp, qr)
    Ψ_nodes = project(projector, Ψ_qp, qr)

    VTKGridFile("hyperelasticity", dh.grid) do vtk
        write_node_data(vtk, evaluate_at_grid_nodes(dh, u, :u), "u") # fluctuation displacement

        write_projection(vtk, projector, F_nodes, "F")
        write_projection(vtk, projector, P_nodes, "P")
        write_projection(vtk, projector, Ψ_nodes, "Psi")

        write_cell_data(vtk, material_number, "Mat")
    end
    return
end