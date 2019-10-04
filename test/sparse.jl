using PolyChaos, Test

function testfunction01(x::Array{Float64,2},deg::Int64,dataset::Tuple)
   op = whichpolynomial(dataset,deg)
   ops = [op]
   mop = MultiOrthoPoly(ops,deg)
   return 0*1 .+ 0.1*evaluate([6],x,mop) .+ 2.0*evaluate([2],x,mop) .+ 10.0*evaluate([3],x,mop) .+ 5.0*evaluate([5],x,mop)
end

Nsamples, deg, numu01   = 5000, 20, 1;
X01 = rand(Nsamples,numu01);
name01 = JacobiOrthoPoly;
dataset01 = (name01,[4.3,10.0]);

coef01 = sort([5.0, 10.0, 2.0, 0.1])
ind01 = sort([[6],[2],[3],[5]])
a01 = testfunction01(X01,deg,dataset01);
@testset "orthosparseMulti test01" begin
   coeff_1,ind_1 = orthosparse(a01,X01,dataset01,deg,numu01)
   @test  sort(ind_1) == sort(ind01)
   @test  isapprox(sort(coeff_1), sort(coef01), atol=1e-3)
end

###############################################################################
Nsamples, deg, numu3   = 5000, 20, 3;
X3 = rand(Nsamples,numu3);
name1 = JacobiOrthoPoly;
dataset1 = (name1,[4.3,10.0]);
name2 = LegendreOrthoPoly;
dataset2 = name2;
name3 = JacobiOrthoPoly;
dataset3 = (name3,[4.3,10.0]);

arr_mixed = [dataset1,dataset2,dataset3];

function testfunction2(x::Array{Float64,2},deg::Int64,dataset::Vector{<:Any})
   op1 = whichpolynomial(dataset[1],deg)
   op2 = whichpolynomial(dataset[2],deg)
   op3 = whichpolynomial(dataset[3],deg)
   ops = [op1,op2,op3];
   mop = MultiOrthoPoly(ops,deg)
   return 0*1 .+ 0.1*evaluate([1,4,1],x,mop) .+ 2.5*evaluate([2,3,2],x,mop) .+ 10.0*evaluate([1,1,5],x,mop) .+ 5.0*evaluate([5,1,0],x,mop)
end

coef3 = sort([5.0, 10.0, 2.5, 0.1])
ind3 = sort([[1,4,1],[2,3,2],[1,1,5],[5,1,0]])
b_mixed = testfunction2(X3,deg,arr_mixed)
@testset "orthosparseMulti test2 mixed polynomials" begin
   coeff_mixed,ind_mixed = orthosparse(b_mixed,X3,arr_mixed,deg,numu3)
   @test  sort(ind_mixed) == sort(ind3)
   @test  isapprox(sort(coeff_mixed), sort(coef3), atol=1e-3)
end

###############################################################################
Nsamples, deg, numu3   = 5000, 20, 3;
X3 = rand(Nsamples,numu3);
name1 = JacobiOrthoPoly;
dataset1 = (name1,[4.3,10.0]);
name2 = JacobiOrthoPoly;
dataset2 = (name2,[4.3,10.0]);
name3 = JacobiOrthoPoly;
dataset3 = (name3,[4.3,10.0]);

arr3 = [dataset1,dataset2,dataset3];

function testfunction3(x::Array{Float64,2},deg::Int64,dataset::Vector{<:Any})
   op1 = whichpolynomial(dataset[1],deg)
   op2 = whichpolynomial(dataset[2],deg)
   op3 = whichpolynomial(dataset[3],deg)
   ops = [op1,op2,op3];
   mop = MultiOrthoPoly(ops,deg)
   return 0*1 .+ 0.1*evaluate([1,4,1],x,mop) .+ 2.5*evaluate([2,3,2],x,mop) .+ 10.0*evaluate([1,1,5],x,mop) .+ 5.0*evaluate([5,1,0],x,mop)
end

coef3 = sort([5.0, 10.0, 2.5, 0.1])
ind3 = sort([[1,4,1],[2,3,2],[1,1,5],[5,1,0]])
c3 = testfunction3(X3,deg,arr3)
@testset "orthosparseMulti test3 similar polynomials" begin
   coeff_3,ind_3 = orthosparse(c3,X3,arr3,deg,numu3)
   @test  sort(ind_3) == sort(ind3)
   @test  isapprox(sort(coeff_3), sort(coef3), atol=1e-3)
end
