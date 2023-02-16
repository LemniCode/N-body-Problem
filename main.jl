#include("./simulator.jl")
using Plots
using LinearAlgebra
#using Parameters

plotlyjs()



# Struct to hold the data in the simulation
mutable struct Bodies
    m :: Array{Float64}
    vel :: Matrix{Float64}
    q :: Matrix{Float64}
    n :: Int64 # number of bodies
    dims :: Int64 # dimension
end

mutable struct Simulator
    bodies :: Bodies
    dt :: Float64
    G :: Float64
    method :: Function
end

# TODO: 
# Returns the current hamiltonian of the system
function calculate_current_H(s::Simulator)
    KE = 0.0

    for i in 1:s.bodies.n
        KE += 0.5 * s.bodies.m[i]*(norm(s.bodies.vel[i,:])^2)
    end

    PE = computePE(s)

    return KE + PE
end

function computePE(s::Simulator)
    #@unpack_Simulator s
    PE = 0.0

    for i in 2:s.bodies.n
        for j in 1:i-1
            r = norm(s.bodies.q[i,:] - s.bodies.q[j,:])
            PE -= s.bodies.m[i]*s.bodies.m[j]/r
        end
    end

    return s.G*PE
end

# Calculates the current angular momentum
function calculate_current_L(s::Simulator)
    total_L = zeros(s.bodies.dims)
    for i in 1:s.bodies.n
        total_L += cross(s.bodies.q[i, :], s.bodies.vel[i, :]*s.bodies.m[i])
    end

    return total_L
end

function computeForce(s::Simulator)
    force = zeros(s.bodies.n, s.bodies.dims) # (n_bodies, dims)
    
    for i in 2:s.bodies.n
        for j in 1:i-1
            dij = s.bodies.q[i, :] - s.bodies.q[j, :]
            r = norm(dij)

            fij = -s.G*s.bodies.m[i]*s.bodies.m[j]/(r*r*r)*dij 

            force[i,:] += fij
            force[j,:] -= fij
        end
    end
    
    return force

end






# Runs a step of the simulation
function step!(s::Simulator)
    s.bodies.vel, s.bodies.q = s.method(s)
end

# Runs N steps of the simulation
function simulate(s::Simulator, n::Int64)
    for _ in 1:n
        step(s)
    end
end


# SIMULATION METHODS

# euler method
function euler(s::Simulator)

    force = computeForce(s)

    nq = s.bodies.q + s.dt*s.bodies.vel
    nvel = s.bodies.vel + force./s.bodies.m*s.dt

    return nvel, nq
end


# euler-b method
function euler_b(s::Simulator)

    force = computeForce(s)

    nvel = s.bodies.vel + force./s.bodies.m*s.dt
    nq = s.bodies.q + s.dt*nvel

    return nvel, nq
end

# Stormer-Verlet method
function stormer_verlet(s::Simulator)

    force = computeForce(s)
    vel_half = s.bodies.vel + force./s.bodies.m * (s.dt/2)
    nq = s.bodies.q + s.dt*vel_half
    s.bodies.q = nq
    force = computeForce(s)
    nvel = vel_half + force./s.bodies.m * (s.dt/2)
    return nvel, nq
end


# Calculates (in 3d space) the velocity of the moon 
# of a planet at a certain distance from said planet
function make_moon3d(planet_mass, G, planet_pos, moon_pos, planet_vel)
    v = copy(planet_vel) # Copy
    # Add a orthogonal component
    dist = moon_pos-planet_pos
    ov = normalize(cross(dist, v))
    ov *= sqrt(G*planet_mass/norm(dist))
    v += ov
    return v
end

# Main function
function overlap_L()

    println("N Body Problem")

    println("N Body Problem")

    N_bodies = 3
    dims = 3 # 2 dimensions
    dt = 100
    N = 3000
    G = 0.00029591

    m = zeros(N_bodies)
    init_q = zeros(N_bodies, dims)
    init_v = zeros(N_bodies, dims)
    bodyName = ["Sun", "Jupyter", "Saturn", "Uranus", "Neptune", "Pluto"] #"Jupyter's Moon", "Jupyter Moon's Moon", "Saturn", "Saturn's Moon 1", "Saturn's Moon 2"]
    m[1] = 1. # sun
    m[2] = 0.000954786 # jupyter
    m[3] = 0.000285583 # saturn
    #m[4] = 0.0000437273164546 # uranus
    #m[5] = 0.0000517759138449 # neptune
    #m[6] = 1/(1.3e8) # pluto

    init_v = [0.0 0.0 0.0;
              0.00565429 -0.00412490 -0.00190589;
              0.00168318 0.00483525 0.00192462;]
              # 0.00354178 0.00137102 0.00055029;
              # 0.00288930 0.00114527 0.00039677;
              # 0.00276725 -0.00170702 -0.00136504]

    init_q = [0.0 0.0 0.0; 
            -3.5023653 -3.8169847 -1.5507963;
            9.0755314 -3.0458353 -1.6483708;]
            # 8.3101420 -16.2901086 -7.2521278;
            # 11.4707666 -25.7294829 -10.8169456;
            # -15.5387357 -25.2225594 -3.1902382]

    bodies = Bodies(m, init_v, init_q, N_bodies, dims)
    bodies2 = Bodies(m, init_v, init_q, N_bodies, dims)
    sim = Simulator(bodies, dt, G, euler_b)
    sim2 = Simulator(bodies2, dt, G, stormer_verlet)
    
    println("Init hamiltonian: $(calculate_current_H(sim))")
    println("Init Angular Momentum: $(calculate_current_L(sim))")

    error_Ls = zeros(N, 2)
    L0 = calculate_current_L(sim)

    for i in 1:N
        step!(sim)
        step!(sim2)
        error_Ls[i, 1] = norm(calculate_current_L(sim)-L0)
        error_Ls[i, 2] = norm(calculate_current_L(sim2)-L0)
    end

    println("After $N steps hamiltonian: $(calculate_current_H(sim))")
    println("After $N steps Ang. Momentum: $(calculate_current_L(sim))")
    println("After $N steps Ang. Momentum: $(calculate_current_L(sim2))")

 
    # X axis be time
    plt = plot(xaxis="step", yaxis="L")
    plot!(error_Ls[:, 1], label = "euler_b")
    plot!(error_Ls[:, 2], label = "stormer_verlet")
    display(plt)
    
end


function solar_system()
    N_bodies = 2
    dims = 3 # 2 dimensions
    dt = 100
    N = 1000
    G = 0.00029591

    m = zeros(N_bodies)
    init_q = zeros(N_bodies, dims)
    init_v = zeros(N_bodies, dims)
    bodyName = ["Sun", "Jupyter", "Saturn", "Uranus", "Neptune", "Pluto"] #"Jupyter's Moon", "Jupyter Moon's Moon", "Saturn", "Saturn's Moon 1", "Saturn's Moon 2"]
    m[1] = 1. # sun
    m[2] = 0.000954786 # jupyter
    #m[3] = 0.000005 # jupyter moon
    #m[4] = 0.00000005 # jupyter moons moon
    m[3] = 0.000285583 # saturn
    #m[6] = 0.00002 # saturn moon
    #m[7] = 0.00002 # saturn moon 2
    m[4] = 0.0000437273164546 # uranus
    m[5] = 0.0000517759138449 # neptune
    m[6] = 1/(1.3e8) # pluto

    init_v = [0.0 0.0 0.0;
              0.00565429 -0.00412490 -0.00190589;
              #0. 0. 0.;
              #0. 0. 0.;
              0.00168318 0.00483525 0.00192462;
              #0. 0. 0.;
              #0. 0. 0.]
              0.00354178 0.00137102 0.00055029;
              0.00288930 0.00114527 0.00039677;
              0.00276725 -0.00170702 -0.00136504]

    init_q = [0.0 0.0 0.0; 
            -3.5023653 -3.8169847 -1.5507963;
            #-3.5023653 -3.95 -1.5507963;
            #-3.5023653 -3.95 -1.555;
            9.0755314 -3.0458353 -1.6483708;
            #9.1355314 -3.0458353 -1.6483708;
            #9.0755314 -3.2458353 -1.6483708]
            8.3101420 -16.2901086 -7.2521278;
            11.4707666 -25.7294829 -10.8169456;
            -15.5387357 -25.2225594 -3.1902382]

    #init_v[3, :] = make_moon3d(m[2], G, init_q[2, :], init_q[3, :], init_v[2, :])
    #init_v[4, :] = make_moon3d(m[3], G, init_q[3, :], init_q[4, :], init_v[3, :])
    #init_v[6, :] = make_moon3d(m[5], G, init_q[5, :], init_q[6, :], init_v[5, :])
    #init_v[7, :] = make_moon3d(m[5], G, init_q[5, :], init_q[7, :], init_v[5, :])

    bodies = Bodies(m, init_v, init_q, N_bodies, dims)
    sim = Simulator(bodies, dt, G, stomer_verlet)
    
    println("Init hamiltonian: $(calculate_current_H(sim))")

    positions = zeros(N, N_bodies, dims);

    for i in 1:N
        step!(sim)
        positions[i, :, :] = sim.bodies.q
    end

    println("After $N steps hamiltonian: $(calculate_current_H(sim))")
 
    # X axis be time
    plt = plot(xaxis="step", yaxis="Hamiltonian")
    for i in 1:N_bodies
        #m_str = string(m[i])
        #m_str = m_str[1:min(4, length(m_str))]
        plot!(plt, positions[:, i, 1], positions[:, i, 2], positions[:, i, 3], label=bodyName[i])
    end
    display(plt)
end


function overlap_hamiltonians()
    println("N Body Problem")

    N_bodies = 6
    dims = 3 # 2 dimensions
    dt = 1
    N = 5000
    G = 0.00029591

    m = zeros(N_bodies)
    init_q = zeros(N_bodies, dims)
    init_v = zeros(N_bodies, dims)
    bodyName = ["Sun", "Jupyter", "Saturn", "Uranus", "Neptune", "Pluto"] #"Jupyter's Moon", "Jupyter Moon's Moon", "Saturn", "Saturn's Moon 1", "Saturn's Moon 2"]
    m[1] = 1. # sun
    m[2] = 0.000954786 # jupyter
    #m[3] = 0.000005 # jupyter moon
    #m[4] = 0.00000005 # jupyter moons moon
    m[3] = 0.000285583 # saturn
    #m[6] = 0.00002 # saturn moon
    #m[7] = 0.00002 # saturn moon 2
    m[4] = 0.0000437273164546 # uranus
    m[5] = 0.0000517759138449 # neptune
    m[6] = 1/(1.3e8) # pluto

    init_v = [0.0 0.0 0.0;
              0.00565429 -0.00412490 -0.00190589;
              #0. 0. 0.;
              #0. 0. 0.;
              0.00168318 0.00483525 0.00192462;
              #0. 0. 0.;
              #0. 0. 0.]
               0.00354178 0.00137102 0.00055029;
               0.00288930 0.00114527 0.00039677;
               0.00276725 -0.00170702 -0.00136504]

    init_q = [0.0 0.0 0.0; 
            -3.5023653 -3.8169847 -1.5507963;
            #-3.5023653 -3.95 -1.5507963;
            #-3.5023653 -3.95 -1.555;
            9.0755314 -3.0458353 -1.6483708;
            #9.1355314 -3.0458353 -1.6483708;
            #9.0755314 -3.2458353 -1.6483708]
             8.3101420 -16.2901086 -7.2521278;
             11.4707666 -25.7294829 -10.8169456;
             -15.5387357 -25.2225594 -3.1902382]

    #init_v[3, :] = make_moon3d(m[2], G, init_q[2, :], init_q[3, :], init_v[2, :])
    #init_v[4, :] = make_moon3d(m[3], G, init_q[3, :], init_q[4, :], init_v[3, :])
    #init_v[6, :] = make_moon3d(m[5], G, init_q[5, :], init_q[6, :], init_v[5, :])
    #init_v[7, :] = make_moon3d(m[5], G, init_q[5, :], init_q[7, :], init_v[5, :])

    bodies = Bodies(m, init_v, init_q, N_bodies, dims)
    bodies2 = Bodies(m, init_v, init_q, N_bodies, dims)
    bodies3 = Bodies(m, init_v, init_q, N_bodies, dims)

    sim = Simulator(bodies, 1, G, euler_b)
    sim2 = Simulator(bodies2, 10, G, euler_b)
    sim3 = Simulator(bodies3, 100, G, euler_b)
    
    println("Init hamiltonian: $(calculate_current_H(sim))")

    #positions = zeros(N, N_bodies, dims);
    hamiltonians = zeros(N, 3)
    times = zeros(N)
    H0 = calculate_current_H(sim)
    for i in 1:N
        step!(sim)
        step!(sim2)
        step!(sim3)
        #positions[i, :, :] = sim.bodies.q
        hamiltonians[i, 1] = abs(calculate_current_H(sim)-H0) / abs(H0)
        hamiltonians[i, 2] = abs(calculate_current_H(sim2)-H0) / abs(H0)
        hamiltonians[i, 3] = abs(calculate_current_H(sim3)-H0) / abs(H0)
        times[i] = dt*i
    end

    println("After $N steps hamiltonian: $(calculate_current_H(sim))")
 
    # X axis be time
    plt = plot(xaxis="step", yaxis="Hamiltonian")
    plot!(plt, times, hamiltonians[:, 1], label = "dt = 1");
    plot!(plt, times, hamiltonians[:, 2], label = "dt = 10");
    plot!(plt, times, hamiltonians[:, 3], label = "dt = 100");

    display(plt)
end


@time overlap_L()