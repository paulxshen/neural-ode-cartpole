#=
Code for the preprint "Neural ODE for Reinforcement Learning and Nonlinear Optimal Control: Cartpole Problem Revisited"
https://medium.com/@paulshen_62280/neural-ode-for-reinforcement-learning-and-nonlinear-optimal-control-cartpole-problem-revisited-5408018b8d71

Hosted on Google Colab notebook https://colab.research.google.com/drive/1p2cg_7SNG-YMhlV2mAI_53Xvbhyka4FK?usp=sharing

Paul Shen
pxshen@alumni.stanford.edu
=#

using DiffEqFlux,
    DifferentialEquations,
    Flux,
    Plots,
    Interpolations,
    DataFrames,
    CSV,
    JSON,
    Plots,
    Dates,
    Dierckx,
    FiniteDifferences,
    Optim

# physical params
m = 1 # pole mass kg
M = 2 # cart mass kg
L = 1 # pole length m
g = 9.8 # acceleration constant m/s^2

# map angle to [-pi, pi)
modpi(theta) = mod2pi(theta + pi) - pi

#=
system dynamics derivative

du: du/dt, state vector derivative updated inplace
u: state vector (x, dx, theta, dtheta)
p: parameter function, here lateral force exerted by cart as a fn of time
t: time
=#
function cartpole(du, u, p, t)
  # position (cart), velocity, pole angle, angular velocity
  x, dx, theta, dtheta = u
  force = p(t)

  du[1] = dx
  du[2] =
    (force + m * sin(theta) * (L * dtheta^2 - g * cos(theta))) /
    (M + m * sin(theta)^2)
  du[3] = dtheta
  du[4] =
    (
      -force * cos(theta) - m * L * dtheta^2 * sin(theta) * cos(theta) +
      (M + m) * g * sin(theta)
    ) / (L * (M + m * sin(theta)^2))
end


# neural network controller, here a simple MLP
# inputs: cos(theta), sin(theta) [to avoid discontinuity in theta mod 2pi], theta_dot
# output: cart force
controller = FastChain((x, p) -> x, FastDense(3, 8, tanh), FastDense(8, 1))

# initial neural network weights
pinit = initial_params(controller)

#=
system dynamics derivative with the controller included
=#
function cartpole_controlled(du, u, p, t)
  # controller force response
  force = controller([cos(u[3]), sin(u[3]), u[4]], p)[1]
  du[5] = force

  # plug force into system dynamics
  cartpole(du, u[1:4], t -> force, t)
end

# initial condition
u0 = [0; 0; pi; 0; 0]
tspan = (0.0, 1.0)
N = 50
tsteps = range(tspan[1], length = N, tspan[2])
dt = (tspan[2] - tspan[1]) / N


# set up ODE problem
prob = ODEProblem(cartpole_controlled, u0, tspan, pinit)

# wrangles output from ODE solver
function format(pred)
  x = pred[1, :]
  dx = pred[2, :]

  theta = modpi.(pred[3, :])
  dtheta = pred[4, :]

  # take derivative of impulse to get force
  impulse = pred[5, :]
  tmp = (impulse .- circshift(impulse, 1)) / dt
  force = [tmp[2], tmp[2:end]...]


  return x, dx, theta, dtheta, force
end

# solves ODE
function predict_neuralode(p)
  tmp_prob = remake(prob, p = p)
  solve(tmp_prob, Tsit5(), saveat = tsteps)
end

# loss to minimize as a function of neural network parameters p
function loss_neuralode(p)
  pred = predict_neuralode(p)
  x, dx, theta, dtheta, force = format(pred)
  loss = sum(theta .^ 2) / N + 4theta[end]^2 + dx[end]^2

  return loss, pred
end


i = 0 # training epoch counter
data = 0 # time series of state vector and control signal
# callback function after each training epoch
callback = function (p, l, pred; doplot = true)
  global i += 1

  global data = format(pred)
  x, dx, theta, dtheta, force = data

  # ouput every few epochs
  if i % 50 == 0
    println(l)
    display(plot(tsteps, theta))
    display(plot(tsteps, x))
    display(plot(tsteps, force))
  end

  return false

end

result = DiffEqFlux.sciml_train(
  loss_neuralode,
  pinit,
  ADAM(0.05),
  cb = callback,
  maxiters = 1500,
)

p = result.minimizer

# save model and data
open(io -> write(io, json(p)), "model.json", "w")
open(io -> write(io, json(data)), "data.json", "w")

gr()
x, dx, theta, dtheta, force = data
anim = Animation()

plt=plot(tsteps,[modpi.(theta.+.01),x,force],title=["Angle" "Position" "Force"],layout=(3,1))
display(plt)
savefig(plt,"cartpole_data.png")

for (x, theta) in zip(x, theta)

    cart = [x - 1 x + 1; 0 0]
    pole = [x x + 10*sin(theta); 0 10*cos(theta)]
    plt = plot(
        cart[1, :],
        cart[2, :],
        xlim = (-10, 10),
        ylim = (-10, 10),
        title = "Cartpole",
        linewidth = 3,
    )
    plot!(plt, pole[1, :], pole[2, :], linewidth = 6)

    frame(anim)
end

gif(anim, "cartpole_animation.gif", fps = 10)
