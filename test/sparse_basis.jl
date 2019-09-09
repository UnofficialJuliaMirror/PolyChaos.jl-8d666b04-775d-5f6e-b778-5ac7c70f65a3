using PolyChaos, Test

###### Toy functions
function testfunction0(x::Array{Float64,2},deg::Int64,name::Array{String,1})
   op1 = OrthoPoly(name[1],deg)
   op2 = OrthoPoly(name[2],deg)
   ops = [op1,op2]
   mop = MultiOrthoPoly(ops,deg)
   return 1 .+ 0.1.*evaluate([1,3],x,mop) .+ 9.0.*evaluate([2,2],x,mop) .+ 0.3.*evaluate([0,1],x,mop) .+ 0.4.*evaluate([2,1],x,mop)
end

function testfunction1(x::Array{Float64,2},deg::Int64,name::Array{String,1})
   op1 = OrthoPoly(name[1],deg)
   op2 = OrthoPoly(name[2],deg)
   ops = [op1,op2]
   mop = MultiOrthoPoly(ops,deg)
   return 1 .+ 0.1.*evaluate([1,3],x,mop) .+ 0.2.*evaluate([2,2],x,mop) .+ 0.3.*evaluate([0,1],x,mop) .+ 0.4.*evaluate([2,1],x,mop)
end

function testfunction2(x::Array{Float64,2},deg::Int64,name::Array{String,1})
   op1 = OrthoPoly(name[1],deg)
   op2 = OrthoPoly(name[2],deg)
   ops = [op1,op2]
   mop = MultiOrthoPoly(ops,deg)
   return 0.0*1 .+ 0.3*evaluate([1,4],x,mop) .+ 2.0*evaluate([2,3],x,mop) .+ 5.0*evaluate([0,1],x,mop) .+ 0.5*evaluate([2,1],x,mop)
end

function testfunction3(x::Array{Float64,2},deg::Int64,name::Array{String,1})
   op1 = OrthoPoly(name[1],deg)
   op2 = OrthoPoly(name[2],deg)
   ops = [op1,op2]
   mop = MultiOrthoPoly(ops,deg)
   return 0*1 .+ 0.1*evaluate([1,4],x,mop) .+ 2.0*evaluate([2,3],x,mop) .+ 10.0*evaluate([1,1],x,mop) .+ 5.0*evaluate([5,1],x,mop)
end

###### 3D case
function testfunction4(x::Array{Float64,2},deg::Int64,name::Array{String,1})
   op1 = OrthoPoly(name[1],deg)
   op2 = OrthoPoly(name[2],deg)
   op3 = OrthoPoly(name[3],deg)
   ops = [op1,op2,op3]
   mop = MultiOrthoPoly(ops,deg)
   return 0*1 .+ 0.1*evaluate([1,4,1],x,mop) .+ 2.5*evaluate([2,3,2],x,mop) .+ 10.0*evaluate([1,1,5],x,mop) .+ 5.0*evaluate([5,1,0],x,mop)
end

##### 1D case
function testfunction5(x::Array{Float64,2},deg::Int64,name::Array{String,1})
   op1 = OrthoPoly(name[1],deg)
   ops = [op1]
   mop = MultiOrthoPoly(ops,deg)
   return 0*1 .+ 0.1*evaluate([6],x,mop) .+ 2.0*evaluate([2],x,mop) .+ 10.0*evaluate([3],x,mop) .+ 5.0*evaluate([5],x,mop)
end

##################################

Nsamples, deg, numu1 , numu2, numu3  = 5000, 20, 1, 2, 3;

X1 = rand(Nsamples,numu1)
X2 = rand(Nsamples,numu2)
X3 = rand(Nsamples,numu3)

name0 = ["uniform01","uniform01"]
name1 = ["gaussian","gaussian"]
name2 = ["uniform01","gaussian"]
name3 = ["uniform01","logistic"]
name4 = ["uniform01","logistic","gaussian"]
name5 = ["uniform01"]

###############################
coef0 = sort([1.0, 0.1, 9.0, 0.3, 0.4])
ind0 = sort([[0,0],[1,3],[2,2],[0,1],[2,1]])
t = testfunction0(X2,deg,name0)
@testset "orthosparseMulti test0" begin
   coeff,ind = orthosparse(t,X2,name0,deg,numu2)
   @test  ind0 == sort(ind)
   @test  isapprox(coef0, sort(coeff), atol=1e-3)
end

#############################
coef1 = sort([0.3, 0.4, 0.2, 0.1, 1.0])
ind1 = sort([[0,0],[0,1],[1,3],[2,1],[2,2]])
u = testfunction1(X2,deg,name1)
@testset "orthosparseMulti test1" begin
   coeff,ind = orthosparse(u,X2,name1,deg,numu2)
   @test  ind1 == sort(ind)
   @test  isapprox(coef1, sort(coeff), atol=1e-3)
end

############################

coef2 = sort([ 0.3, 5.0, 2.0, 0.5])
ind2 = sort([[0,1],[1,4],[2,1],[2,3]])
v = testfunction2(X2,deg,name2)
@testset "orthosparseMulti test2" begin
   coeff,ind = orthosparse(v,X2,name2,deg,numu2)
   @test  ind2 == sort(ind)
   @test  isapprox(coef2, sort(coeff), atol=1e-3)
end

###########################
coef3 = sort([5.0, 10.0, 2.0, 0.1])
ind3 = sort([[1,4],[2,3],[1,1],[5,1]])
w = testfunction3(X2,deg,name3)
@testset "orthosparseMulti test3" begin
   coeff,ind = orthosparse(w,X2,name3,deg,numu2)
   @test  ind3 == sort(ind)
   @test  isapprox(coef3, sort(coeff), atol=1e-4)
end

######################### 3D case
coef4 = sort([5.0, 10.0, 2.5, 0.1])
ind4 = sort([[1,4,1],[2,3,2],[1,1,5],[5,1,0]])
b = testfunction4(X3,deg,name4)
@testset "orthosparseMulti test4" begin
   coeff,ind = orthosparse(b,X3,name4,deg,numu3)
   @test  ind4 == sort(ind)
   @test  isapprox(coef4, sort(coeff), atol=1e-3)
end

########################### 1D case
coef5 = sort([5.0, 10.0, 2.0, 0.1])
ind5 = sort([[6],[2],[3],[5]])
c = testfunction5(X1,deg,name5)
@testset "orthosparseMulti test5" begin
   coeff,ind = orthosparse(c,X1,name5,deg,numu1)
   @test  ind5 == sort(ind)
   @test  isapprox(coef5, sort(coeff), atol=1e-3)
end
