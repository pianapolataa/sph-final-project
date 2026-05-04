# NYU CS 421 Numerical Computing SPH project
# Prepared by Dr. Gizem Kayar in April 2026
# To be completed by Spring 2026 CS 421 students

using LinearAlgebra
using CSV
using DataFrames

# -----------------------------
# Parameters
# -----------------------------

 # TO DO 6
const h    = 0.04 
const mass = 1.06
const rho0 = 1000.0
const k    = 3000.0
const mu   = 0.1
const g    = [0.0, -9.8]
const dt   = 2.0e-5
const STEPS = 100000
const SAVE_EVERY = 10
const dx   = h * 0.8

const cell_size = h
const grid_res = Int(ceil(1.0 / cell_size))  # domain [0,1]

# -----------------------------
# Kernels
# -----------------------------

 # TO DO 7
W_poly6(r) = (0 ≤ r ≤ h) ? 4/(pi*h^8)*(h^2 - r^2)^3 : 0.0

function gradW_spiky(rvec)
    r = norm(rvec)
    (0 < r ≤ h) ? -30/(pi*h^5)*(h - r)^2*(rvec/r) : zeros(2)
end

lapW_visc(r) = (0 ≤ r ≤ h) ? 20/(3*pi*h^5)*(h - r) : 0.0

# -----------------------------
# Initialization (corner dam)
# -----------------------------
function init_particles()
    
    pos = Vector{Vector{Float64}}()
    vel = Vector{Vector{Float64}}()

    # Dense block in lower-left corner
    # TO DO 1 and 8
   
    # TODO 1:
    for x in dx : dx : 0.4
        for y in dx : dx : 0.6
            push!(pos, [x, y])       # initial position
            push!(vel, [0.0, 0.0])   # Initial velocity set to zero 
        end
    end



    Np = length(pos)
    rho = zeros(Np)
    P   = zeros(Np)

    return pos, vel, rho, P
end

# -----------------------------
# Grid (spatial hashing)
# -----------------------------
function build_grid(pos)
    grid = Dict{Tuple{Int,Int}, Vector{Int}}()

    for i in eachindex(pos)
        cx = clamp(Int(floor(pos[i][1] / cell_size)), 0, grid_res)
        cy = clamp(Int(floor(pos[i][2] / cell_size)), 0, grid_res)
        key = (cx, cy)

        if haskey(grid, key)
            push!(grid[key], i)
        else
            grid[key] = [i]
        end
    end
    return grid
end

# -----------------------------
# Neighbor search
# -----------------------------
function find_neighbors(pos, grid)
    neighbors = [Int[] for _ in eachindex(pos)]

    # TO DO 2 and 9
    
    # TODO 2:
    for i in eachindex(pos)
        cx = clamp(Int(floor(pos[i][1] / cell_size)), 0, grid_res)
        cy = clamp(Int(floor(pos[i][2] / cell_size)), 0, grid_res)

        # check surrounding 9 cells' hash keys
        for dx_offset in -1:1
            for dy_offset in -1:1
                key = (cx + dx_offset, cy + dy_offset)

                # push other particles in grid[key]
                if haskey(grid, key)
                    for j in grid[key]
                        if i != j 
                            if norm(pos[i] - pos[j]) <= h # smoothing, only oush if within distance h
                                push!(neighbors[i], j) 
                            end
                        end
                    end
                end
            end
        end
    end




    return neighbors
end

# -----------------------------
# Density & Pressure
# -----------------------------
function compute_density!(pos, rho, neighbors)
    for i in eachindex(pos)
        ρ = mass * W_poly6(0.0) #initialize density with self contribution

        # TO DO 3 - compute density via smoothing 
        for j in neighbors[i]
            r = norm(pos[i] - pos[j])
            ρ += mass * W_poly6(r)
        end

        rho[i] = max(ρ, 1e-6)  # prevent division issues
    end
end

compute_pressure!(rho, P) = (P .= max.(k .* (rho .- rho0), 0.0))

# -----------------------------
# Forces
# -----------------------------
function compute_forces(pos, vel, rho, P, neighbors)
    forces = [zeros(2) for _ in eachindex(pos)]

    for i in eachindex(pos)
        f_p = zeros(2)
        f_v = zeros(2)

        for j in neighbors[i]
            rij = pos[i] - pos[j]
            r = norm(rij)

            
             # TO DO 4 - compute pressure force and viscosity force

            # # Symmetric pressure force (stable)
            f_p += -mass * (P[i] + P[j]) / (2.0 * rho[i] * rho[j]) * gradW_spiky(rij)     

            # Viscosity
            f_v += mu * mass * (vel[j] - vel[i]) / rho[j] * lapW_visc(r)
        end

        forces[i] = f_p + f_v + g #gravity also added
    end

    return forces
end

# -----------------------------
# Integration (Euler-Cromer)
# -----------------------------
function integrate!(pos, vel, accel)
    # TO DO 5 and 10
    wall_damping = 0.5  
    floor_damping = 0.2 

    for i in eachindex(pos)
        vel[i] += accel[i] * dt # update velocity
        pos[i] += vel[i] * dt # update position

        if pos[i][1] < 0.0 # left wall
            pos[i][1] = 0.0
            vel[i][1] *= -wall_damping
        elseif pos[i][1] > 1.0 # right wall
            pos[i][1] = 1.0
            vel[i][1] *= -wall_damping
        end

        if pos[i][2] < 0.0 # floor
            pos[i][2] = 0.0
            vel[i][2] *= -floor_damping
        end
        
    end
    
end

function xsph!(vel, pos, rho, neighbors)
    ε = 0.5
    newvel = deepcopy(vel)

    for i in eachindex(pos)
        corr = zeros(2)
        for j in neighbors[i]
            corr += mass * (vel[j] - vel[i]) / rho[j] *
                    W_poly6(norm(pos[i] - pos[j]))
        end
        newvel[i] += ε * corr
    end

    for i in eachindex(vel)
        vel[i] = newvel[i]
    end
end

# -----------------------------
# CSV output
# -----------------------------
function save_csv(pos, vel, rho, P, step)
    outdir = "output"
    isdir(outdir) || mkdir(outdir)

    filename = joinpath(outdir, "sph_$(lpad(step,6,'0')).csv")

    open(filename, "w") do io
        for p in pos
            println(io, "$(p[1]),$(p[2])")
        end
    end
end

# -----------------------------
# Main loop
# -----------------------------
function run()
    pos, vel, rho, P = init_particles()

    for step in 1:STEPS
        grid = build_grid(pos)
        neighbors = find_neighbors(pos, grid)

        compute_density!(pos, rho, neighbors)
        compute_pressure!(rho, P)
        accel = compute_forces(pos, vel, rho, P, neighbors)
        integrate!(pos, vel, accel)

        if step % SAVE_EVERY == 0
            save_csv(pos, vel, rho, P, step)
        end
    end

    println("Done (corner breaking dam).")
end

run()