# Solver
function solve(F_macro, mp_i, mp_m, mesh_filename; output=:false)
    # Read mesh from file
    grid = togrid(mesh_filename)

    # Finite element base
    dim = 2
    ip = Lagrange{RefTriangle, 1}()^dim
    qr = QuadratureRule{RefTriangle}(1)
    cv = CellValues(qr, ip)

    # DofHandler
    dh = DofHandler(grid)
    add!(dh, :u, ip) # Add a displacement field
    close!(dh)

    # Add periodic boundary conditions
    ch = ConstraintHandler(dh)
    periodic = PeriodicDirichlet(:u, ["left"=>"right", "bottom"=>"top"], [1,2])
    add!(ch, periodic)
    close!(ch)
    Ferrite.update!(ch, 0.0)

    # Create material states
    n_qp = getnquadpoints(cv)
    states = [[MaterialState() for _ in 1:n_qp] for _ in 1:getncells(grid)]

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    un = zeros(_ndofs) # previous solution vector
    u = zeros(_ndofs)
    Δu = zeros(_ndofs)
    ΔΔu = zeros(_ndofs)
    apply!(un, ch)

    # Create sparse matrix and residual vector
    K = allocate_matrix(dh, ch)
    g = zeros(_ndofs)

    # Perform Newton iterations
    newton_itr = -1
    NEWTON_TOL = 1.0e-8
    NEWTON_MAXITER = 30
    prog = ProgressMeter.ProgressThresh(NEWTON_TOL; desc = "Solving:")

    while true
        newton_itr += 1
        # Construct the current guess
        u .= un .+ Δu
        # Compute residual and tangent for current guess
        assemble_global!(K, g, dh, cv, F_macro, mp_i, mp_m, u, states)
        # Apply boundary conditions
        apply!(K, g, ch)
        # Compute the residual norm and compare with tolerance
        normg = norm(g) # alternative convergence criterium?? incremental...

        ProgressMeter.update!(prog, normg; showvalues = [(:iter, newton_itr)])

        if normg < NEWTON_TOL
            break
        elseif newton_itr > NEWTON_MAXITER
            error("Reached maximum Newton iterations, aborting")
        end

        # Compute increment using conjugate gradients
        IterativeSolvers.cg!(ΔΔu, K, g; maxiter = 1000)
        #ΔΔu = K\g # not better or worse compared to cg...

        apply!(ΔΔu, ch)
        Δu .-= ΔΔu
    end

    # Calculate averages
    P_aver, Ψ_aver = volaver(cv, dh, states, dim)

    # Write output as vtk file
    if(output)
        ip_h = Lagrange{RefTriangle, 1}() # helper for L2projection
        writeoutput(ip_h, qr, dh, u, states, dim)
    end

    return P_aver, Ψ_aver
end

# Macroscopic tangent calculation
function calculate_tangent(mp_i, mp_m, mesh_filename)
    F_macro_11 = SymmetricTensor{2, 2}([1.0 0.0; 0.0 0.0])
    F_macro_22 = SymmetricTensor{2, 2}([0.0 0.0; 0.0 1.0])
    F_macro_12 = Tensor{2, 2}([0.0 1.0; 0.0 0.0])
    F_macro_21 = Tensor{2, 2}([0.0 0.0; 1.0 0.0])
    P_aver_11, _ = solve(F_macro_11, mp_i, mp_m, mesh_filename);
    P_aver_22, _ = solve(F_macro_22, mp_i, mp_m, mesh_filename);
    P_aver_12, _ = solve(F_macro_12, mp_i, mp_m, mesh_filename);
    P_aver_21, _ = solve(F_macro_21, mp_i, mp_m, mesh_filename);
    ∂F∂P_macro = SymmetricTensor{4, 2}() do i, j, k, l
        if(k == l == 1)
            P_aver_11[i,j]
        elseif(k == l == 2)
            P_aver_22[i,j]
        elseif(k==1 && l==2)
            P_aver_12[i,j]
        else
            P_aver_21[i,j]
        end
    end
    return ∂F∂P_macro 
end