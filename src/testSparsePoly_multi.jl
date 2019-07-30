using PolyChaos

include("sparse_basis.jl")
include("multiIndices.jl")

Nsamples, deg, numu, name = 1000 , 9, 2,["uniform01","uniform01"]

X = randn(Nsamples,numu)

function testfunction(x::Array{Float64,2},deg::Int64)
   op1 = OrthoPoly(name[1],deg)
   op2 = OrthoPoly(name[2],deg)
   # op3 = OrthoPoly(name[3],deg)
   ops = [op1,op2]
   # ops = [op1,op2,op3]
   mop = MultiOrthoPoly(ops,deg)
   return 1 .+ evaluate([1,3],x,mop) .+ evaluate([2,2],x,mop) .+ 0.2*evaluate([0,1],x,mop) .+ 0.9*evaluate([2,1],x,mop)
   # return 1 .+ evaluate([1,1],x,mop) .+  evaluate([0,1],x,mop) .+ 0.1*evaluate([1,2],x,mop)
   # return 0*1 .+ evaluate([1,3,1],x,mop) .+  evaluate([0,1,2],x,mop) .+ 0.1*evaluate([2,1,1],x,mop)
end
Y = testfunction(X,deg);
m = Array{Array{Int64,1},1}();
m = orthosparseMulti(Y,X,name,deg,numu,0.00000001,0.00000001,0.9999999999)
print("returns ", m)
