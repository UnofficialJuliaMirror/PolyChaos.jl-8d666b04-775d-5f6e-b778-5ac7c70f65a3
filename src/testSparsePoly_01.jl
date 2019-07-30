using PolyChaos
include("sparse_basis.jl")


N = 1000
x = rand(N); sort!(x);

# name = "uniform01"
name = "legendre"
# name = "gaussian"
ε1, ε2, Qacc, pmax = 1e-5, 1e-5,1- 1e-9, 15
op_legendre = OrthoPoly(name,pmax);
# op_uniform01 = OrthoPoly(name,pmax);
# op_gaussian = OrthoPoly(name,pmax);

function d(t)
    return 10.0 .+ 0.5*evaluate(3,t,op_legendre) .+ evaluate(2,t,op_legendre)
    # return 10.0 .+ evaluate(3,t,op_uniform01) .+ 3.0*evaluate(2,t,op_uniform01) .+ 0.9*evaluate(8,t,op_uniform01)
    # return 10.0 .+ 5.0.*evaluate(4,t,op_gaussian) .+ evaluate(5,t,op_gaussian)
end
y = d.(x)
inds = orthosparse(y,x,name,pmax,ε_forward=ε1,ε_backward=ε2,accuracy=Qacc)

print("returns ", inds)
