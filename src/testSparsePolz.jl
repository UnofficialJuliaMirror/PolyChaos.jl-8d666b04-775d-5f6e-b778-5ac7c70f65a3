using PolyChaos, LinearAlgebra
include("sparse_basis.jl")
include("multiIndices.jl")
#function c(x)
#       return 1*(x^3 - 3*x) + (x^4 - 6*x^2 + 3.0) + (x^6 - 15*x^4 + 45*x^2 - 15) + 0*3.0
#end

#function b(x)
#    return 0*1 + 1*(x^2 - 1) + 1*(x^10 - 45*x^8 + 630*x^6 - 3150*x^4 + 4725*x^2 - 945)
#end

#function a(x)
#    return 0*1 + 0*(x^2 - 1) + 0*(x^3 - 3*x) + 0*(x^4 - 6*x^2 + 3.0) + 0*(x^6 - 15*x^4 + 45*x^2 - 15) + 1*(x^10 - 45*x^8 + 630*x^6 - 3150*x^4 + 4725*x^2 - 945)
#end

#y = h.(x)
#y = b.(x)
#y = a.(x)







N = 100
x = rand(N); sort!(x);#display(x)

name = "uniform01"
#name = "legendre"
#name = "logistic"
#name = "jacobi"
ε1, ε2, Qacc, pmax = 1e-5, 1e-8, 1 - 1e-8, 10
op_legendre = OrthoPoly(name,pmax);
#op_gaussian = OrthoPoly(name,pmax);
#op_logistic = OrthoPoly(name,pmax);
function d(t)
    #return 10.0 .+ 5.0.*evaluate(4,t,op_legendre) .+ evaluate(5,t,op_legendre)
    #return 0 + 1*evaluate(4,t,op_legendre) + evaluate(1,t,op_legendre)
    return 1.0 .+ 6.0*evaluate(3,t,op_legendre) .+ 1.0*evaluate(5,t,op_legendre)
    #return 0.5.*evaluate(4,t,op_legendre) .+ 0.2.*evaluate(5,t,op_legendre)
    #return 10.0 .+ 5.0.*evaluate(4,t,op_gaussian) .+ evaluate(5,t,op_gaussian)
    #return 5.0 .+ 0.5.*evaluate(4,t,op_gaussian) .+ 0.1.*evaluate(5,t,op_gaussian)
#    return 5.0 .- 0.3.*evaluate(4,t,op_gaussian) .+ 0.1.*evaluate(5,t,op_gaussian)
    #return 0.5.*evaluate(4,t,op_gaussian) .+ 0.2.*evaluate(5,t,op_gaussian)
    #return 10.0 .+ 5.0.*evaluate(4,t,op_logistic) .+ evaluate(5,t,op_logistic)
    #return 5.0 .+ 0.5.*evaluate(4,t,op_logistic);#.+ evaluate(5,t,op_logistic)
    #return 1.0 .- 0.5.*evaluate(4,t,op_logistic) .+ 0.2.*evaluate(5,t,op_logistic)
    #return 0.5.*evaluate(4,t,op_logistic) .+ 0.2.*evaluate(5,t,op_logistic)
end
y = d.(x)
inds = orthosparse(y,x,name,pmax,ε_forward=ε1,ε_backward=ε2,accuracy=Qacc)

print("returns ", inds)








## how to generate the index sets
# nunc = 2        #   number of uncertainties
# deg = 6         #   maximum total degree
# j = 2           #   interaction order
# p = 6            #   desired degree
# I_p = []
# # for j in 1:2
# #    push!(I_p,calculateMultiIndices_interaction(nunc,deg,j,p))
# # end
# I_p = calculateMultiIndices_interaction(nunc,deg,j,p)
# for i in I_p
#    display(i)
# end
#
#Nsamples, deg, numu = 1000 , 20, 2
#X = randn(Nsamples,numu)

#function etc(v::SubArray{Float64,1})
#    return 1 + v[1] + v[2] + v[1]*v[2] + v[1]*v[1] - 1
#end

#function q(x::Array{Float64,2})
#    G = Float64[]
#    for r in eachrow(x)
#        push!(G,etc(r))
#    end
#    return G
#end
#Y = q(X)

#m = orthosparseMulti(Y,X,"gaussian",deg,numu,0.01,0.01,0.9999999999)
#print("returns ", m)
