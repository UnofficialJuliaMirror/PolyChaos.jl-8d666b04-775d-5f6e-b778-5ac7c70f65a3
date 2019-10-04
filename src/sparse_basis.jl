export orthosparse, whichpolynomial


function regress(M,y)
  c = pinv(M)*y
  c, M*c
end

function lack_of_fit(y::Array{Float64,1},ymod::Array{Float64,1})
    lof = sum(δy^2 for δy in y - ymod) / sum(δy^2 for δy in y .- mean(y))
    lof > 1 ? 1 : lof
end

function matrix_to_vector(a::Array{Int64,2})
    return copy.(eachrow(a))
end

function vector_to_matrix(b::Array{Array{Int64,1},1})
    return reduce(vcat, transpose.(b))
end

function whichpolynomial(t::Tuple, deg::Int)
    return length(t) == 1 ? t[1](deg) : t[1](deg, t[2]...);
end


function whichpolynomial(t, deg::Int)
    return t(deg);
end


function generateI_p(numu::Int64,pthorder::Int64)
    setI_p = Array{Array{Int64,1},1}();
    indices = calculateMultiIndices(numu,pthorder);
    indices = matrix_to_vector(indices);
    for k in indices
        sum(k) == pthorder ? push!(setI_p,Vector(k)) : Vector(k)
    end
    return setI_p;
end

function forwardstep(numu,p,A,x,y,R,ε_forward,mop)
    I_p = generateI_p(numu,p);
    typeof(R) == Int64 ? Float64(R) : Float64(R)
    typeof(A) == Array{Int64,2} ? A = matrix_to_vector(A) : A
    for j in I_p
        push!(A,Vector(j))
        A = vector_to_matrix(A);
        Φ1 = evaluate(A,x,mop)
        Φ1 = transpose(Φ1)
        a_hat, ymod = regress(Φ1,y)
        R2_new = 1 - lack_of_fit(y,ymod)
        A = matrix_to_vector(A);
        # if R2_new is significantly better, then push to the basis/Index set
        if R2_new == 1.0 || abs(R2_new - R) > ε_forward
            R = R2_new
        else
            pop!(A)
        end
    end
    return R, A;
end

function backwardstep(numu,p,A,x,y,R,ε_backward,mop)
    typeof(R) == Int64 ? Float64(R) : Float64(R)
    typeof(A) == Array{Array{Int64,1},1} ? A : A = matrix_to_vector(A)
    A_plus = Array{Array{Int64,1},1}()
    A_a = A[[sum(t) == p for t in A]]; #set of pth order basis
    A_b = A[[sum(t) < p for t in A]]; #set of all basis of order less than p
    if A_a != []
        for s in A_b
            A_plus = A[[t != s for t in A]];
            A_plus = vector_to_matrix(A_plus);
            Φ2 = evaluate(A_plus,x,mop);
            Φ2 = transpose(Φ2);
            a_tmp, ymod = regress(Φ2,y);
            r2_new = 0.0;
            r2 = 1 - lack_of_fit(y,ymod);
            A_plus = matrix_to_vector(A_plus);
            if abs(R - r2) < ε_backward || r2 == 1.0
                A = A_plus;
            else
                push!(A_plus,Vector(s))
            end
        end
        A = A_plus
    end
    A = vector_to_matrix(A);
    return  A
end

# function orthosparse(y::Vector{Float64},x::Vector{Float64},name::String,p_max::Int,numu::Int64;ε_forward::Float64=1e-12 , ε_backward::Float64=1e-12 , accuracy::Float64=1.0-1e-15)
#     p_index = 1:p_max
#     op = OrthoPoly(name,p_max)
#     y_bar = mean(y)
#     A, A_plus = [0], []
#     Φ0 = evaluate(A,x,op)
#     a_hat, ymod = regress(Φ0,y)
#     R2 = 1 - lack_of_fit(y,ymod)
#     for p in p_index
#         p >= p_max && break
#
#         ##### FORWARD STEP
#         push!(A,p)
#         Φ1 = evaluate(A,x,op)
#         a_hat, ymod = regress(Φ1,y)
#         R2_new = 1 - lack_of_fit(y,ymod)
#         # if R2_new is significantly better, then push to the basis
#         if abs(R2_new - R2) >= ε_forward
#             R2 = R2_new # do nothing
#         elseif R2_new ==1.0
#             R2 = R2_new ## to check if R2 is already 1.0
#         else
#             filter!(x -> x != p, A)
#         end
#         A_plus = A
#
#         ##### Backward STEP
#         if A_plus[end] == p
#           for b in A_plus[1:end-1]
#             Φ2 = evaluate(filter(x -> x ≠ b, A_plus),x,op)
#             a_tmp, ymod = regress(Φ2,y)
#             r2 = 1 - lack_of_fit(y,ymod)
#             if abs(R2 - r2) < ε_backward
#               filter!(x -> x ≠ b, A_plus)
#             end
#           end
#           A = A_plus
#         end
#
#         ##### CHECK TERMINATION CRITERION
#         if length(A) != 0
#           Φ3 = evaluate(A,x,op)
#           a_tmp, ymod = regress(Φ3,y)
#           R2 = 1 - lack_of_fit(y,ymod)
#           if R2 >= accuracy
#             println("R2: ", R2)
#             println("coeff: ", a_tmp)
#             println("indices: ", A)
#             println("Accuracy achieved. Breaking.\n")
#             return a_tmp, A
#           end
#         end
#         #####
#     end
#     error("Algorithm terminated early; perhaps a pathological problem was provided.")
# end

function orthosparse(y::Vector{Float64},x::Matrix{Float64},dataset::Tuple,p_max::Int,numu::Int64; ε_forward::Float64=1e-12 , ε_backward::Float64=1e-12 , accuracy::Float64=1.0-1e-15)
    p_index = 1:p_max
    ##### Generates Multivariate ortho Polynomials
    ops = Vector{AbstractOrthoPoly}();
    push!(ops,whichpolynomial(dataset,p_max))
    mop = MultiOrthoPoly(ops,p_max) # Multivariate Orthogonal Polynomials

    A = Array{Array{Int64,1},1}(); # Index set
    # A_plus = Array{Array{Int64,1},1}(); # Index set for forward step
    push!(A,vec(zeros(Int64,1,numu)));
    A = vector_to_matrix(A);
    R2 = 0.0;
    Φ0 = evaluate(A,x,mop)
    Φ0 = transpose(Φ0)
    a_tmp,ymod = regress(Φ0,y)
    R2 = 1 - lack_of_fit(y,ymod)
    A = matrix_to_vector(A);
    for p in p_index
        p >= p_max && break

        ####### FORWARD STEP
        ##### Generating the set multiindecies of pth order
        # I_p = Array{Array{Int64,1},1}();
        # # for j in j_index
        # #     indices_set = calculateMultiIndices_interaction(numu,p_max,j,p)
        # #     for k in indices_set
        # #         push!(I_p,Vector(k))
        # #     end
        # # end
        # I_p = generateI_p(numu,p);
        # ####
        # for j in I_p
        #     push!(A,Vector(j))
        #     A = vector_to_matrix(A);
        #     Φ1 = evaluate(A,x,mop)
        #     Φ1 = transpose(Φ1)
        #     a_hat, ymod = regress(Φ1,y)
        #     R2_new = 1 - lack_of_fit(y,ymod)
        #     A = matrix_to_vector(A);
        #     # if R2_new is significantly better, then push to the basis/Index set
        #     if R2_new == 1.0 # do nothing
        #         R2 = R2_new
        #     elseif abs(R2_new - R2) > ε_forward # do nothing
        #         R2 = R2_new
        #     else
        #         pop!(A)
        #     end
        # end
        R2, A = forwardstep(numu,p,A,x,y,R2,ε_forward,mop);

        ##### Backward STEP
        # A_a = A[[sum(x) == p for x in A]]; #set of pth order basis
        # A_b = A[[sum(x) < p for x in A]]; #set of all basis of order less than p
        # if A_a != []
        #     for s in A_b
        #         A_plus = A[[x != s for x in A]];
        #         A_plus = vector_to_matrix(A_plus);
        #         Φ2 = evaluate(A_plus,x,mop);
        #         Φ2 = transpose(Φ2);
        #         a_tmp, ymod = regress(Φ2,y);
        #         r2_new = 0.0;
        #         r2 = 1 - lack_of_fit(y,ymod);
        #         A_plus = matrix_to_vector(A_plus);
        #         if abs(R2 - r2) < ε_backward || r2 == 1.0
        #             A = A_plus;
        #         else
        #             push!(A_plus,Vector(s))
        #         end
        #     end
        #     A = A_plus
        # end
        # A = vector_to_matrix(A);
        A = backwardstep(numu,p,A,x,y,R2,ε_backward,mop)

        ##### CHECK TERMINATION CRITERION
        if size(A)[1]  != 0
            Φ3 = evaluate(A,x,mop)
            Φ3 = transpose(Φ3)
            a_tmp, ymd = regress(Φ3,y)
            R2 = 1 - lack_of_fit(y,ymd)
            A = matrix_to_vector(A);
            if R2 >= accuracy || isapprox(R2,accuracy,atol=1e-15)
                temp_ind = findall(x->x>1e-8,a_tmp)
                A = A[temp_ind]
                a_tmp = a_tmp[temp_ind]
                println("R2: ", R2)
                println("coeff: ", a_tmp)
                println("indices: ", A)
                println("Accuracy achieved. Breaking\n")
                return a_tmp,A
            end
        end
        ######
    end
    error("Algorithm terminated early; perhaps a pathological problem was provided.")
end

function orthosparse(y::Vector{Float64},x::Matrix{Float64},dataset::Array{<:Tuple,1},p_max::Int,numu::Int64; ε_forward::Float64=1e-12 , ε_backward::Float64=1e-12 , accuracy::Float64=1.0-1e-15)
    p_index = 1:p_max
    ##### Generates Multivariate ortho Polynomials
    ops = Vector{AbstractOrthoPoly}();
    for j in dataset
        push!(ops,whichpolynomial(j,p_max))
    end
    mop = MultiOrthoPoly(ops,p_max) # Multivariate Orthogonal Polynomials

    A = Array{Array{Int64,1},1}(); # Index set
    # A_plus = Array{Array{Int64,1},1}(); # Index set for forward step
    push!(A,vec(zeros(Int64,1,numu)));
    A = vector_to_matrix(A);
    R2 = 0.0;
    Φ0 = evaluate(A,x,mop)
    Φ0 = transpose(Φ0)
    a_tmp,ymod = regress(Φ0,y)
    R2 = 1 - lack_of_fit(y,ymod)
    A = matrix_to_vector(A);
    for p in p_index
        p >= p_max && break

        ####### FORWARD STEP
        ##### Generating the set multiindecies of pth order
        # I_p = Array{Array{Int64,1},1}();
        # # for j in j_index
        # #     indices_set = calculateMultiIndices_interaction(numu,p_max,j,p)
        # #     for k in indices_set
        # #         push!(I_p,Vector(k))
        # #     end
        # # end
        # I_p = generateI_p(numu,p);
        # ####
        # for j in I_p
        #     push!(A,Vector(j))
        #     A = vector_to_matrix(A);
        #     Φ1 = evaluate(A,x,mop)
        #     Φ1 = transpose(Φ1)
        #     a_hat, ymod = regress(Φ1,y)
        #     R2_new = 1 - lack_of_fit(y,ymod)
        #     A = matrix_to_vector(A);
        #     # if R2_new is significantly better, then push to the basis/Index set
        #     if R2_new == 1.0 # do nothing
        #         R2 = R2_new
        #     elseif abs(R2_new - R2) > ε_forward # do nothing
        #         R2 = R2_new
        #     else
        #         pop!(A)
        #     end
        # end
        R2, A = forwardstep(numu,p,A,x,y,R2,ε_forward,mop);

        ##### Backward STEP
        # A_a = A[[sum(x) == p for x in A]]; #set of pth order basis
        # A_b = A[[sum(x) < p for x in A]]; #set of all basis of order less than p
        # if A_a != []
        #     for s in A_b
        #         A_plus = A[[x != s for x in A]];
        #         A_plus = vector_to_matrix(A_plus);
        #         Φ2 = evaluate(A_plus,x,mop);
        #         Φ2 = transpose(Φ2);
        #         a_tmp, ymod = regress(Φ2,y);
        #         r2_new = 0.0;
        #         r2 = 1 - lack_of_fit(y,ymod);
        #         A_plus = matrix_to_vector(A_plus);
        #         if abs(R2 - r2) < ε_backward || r2 == 1.0
        #             A = A_plus;
        #         else
        #             push!(A_plus,Vector(s))
        #         end
        #     end
        #     A = A_plus
        # end
        # A = vector_to_matrix(A);
        A = backwardstep(numu,p,A,x,y,R2,ε_backward,mop)


        ##### CHECK TERMINATION CRITERION
        if size(A)[1]  != 0
            Φ3 = evaluate(A,x,mop)
            Φ3 = transpose(Φ3)
            a_tmp, ymd = regress(Φ3,y)
            R2 = 1 - lack_of_fit(y,ymd)
            A = matrix_to_vector(A);
            if R2 >= accuracy || isapprox(R2,accuracy,atol=1e-15)
                temp_ind = findall(x->x>1e-4,a_tmp)
                A = A[temp_ind]
                a_tmp = a_tmp[temp_ind]
                println("R2: ", R2)
                println("coeff: ", a_tmp)
                println("indices: ", A)
                println("Accuracy achieved. Breaking\n")
                return a_tmp,A
            end
        end
        ######
    end
    error("Algorithm terminated early; perhaps a pathological problem was provided.")
end

function orthosparse(y::Vector{Float64},x::Matrix{Float64},dataset::Array{Any,1},p_max::Int,numu::Int64; ε_forward::Float64=1e-12 , ε_backward::Float64=1e-12 , accuracy::Float64=1.0-1e-15)
    p_index = 1:p_max
    ##### Generates Multivariate ortho Polynomials
    ops = Vector{AbstractOrthoPoly}();
    for j in dataset
        push!(ops,whichpolynomial(j,p_max))
    end
    mop = MultiOrthoPoly(ops,p_max) # Multivariate Orthogonal Polynomials

    A = Array{Array{Int64,1},1}(); # Index set
    # A_plus = Array{Array{Int64,1},1}(); # Index set for forward step
    push!(A,vec(zeros(Int64,1,numu)));
    A = vector_to_matrix(A);
    R2 = 0.0;
    Φ0 = evaluate(A,x,mop)
    Φ0 = transpose(Φ0)
    a_tmp,ymod = regress(Φ0,y)
    R2 = 1 - lack_of_fit(y,ymod)
    A = matrix_to_vector(A);
    for p in p_index
        p >= p_max && break

        ####### FORWARD STEP
        ##### Generating the set multiindecies of pth order
        # I_p = Array{Array{Int64,1},1}();
        # # for j in j_index
        # #     indices_set = calculateMultiIndices_interaction(numu,p_max,j,p)
        # #     for k in indices_set
        # #         push!(I_p,Vector(k))
        # #     end
        # # end
        # I_p = generateI_p(numu,p);
        # ####
        # for j in I_p
        #     push!(A,Vector(j))
        #     A = vector_to_matrix(A);
        #     Φ1 = evaluate(A,x,mop)
        #     Φ1 = transpose(Φ1)
        #     a_hat, ymod = regress(Φ1,y)
        #     R2_new = 1 - lack_of_fit(y,ymod)
        #     A = matrix_to_vector(A);
        #     # if R2_new is significantly better, then push to the basis/Index set
        #     if R2_new == 1.0 # do nothing
        #         R2 = R2_new
        #     elseif abs(R2_new - R2) > ε_forward # do nothing
        #         R2 = R2_new
        #     else
        #         pop!(A)
        #     end
        # end
        R2, A = forwardstep(numu,p,A,x,y,R2,ε_forward,mop);

        ##### Backward STEP
        # A_a = A[[sum(x) == p for x in A]]; #set of pth order basis
        # A_b = A[[sum(x) < p for x in A]]; #set of all basis of order less than p
        # if A_a != []
        #     for s in A_b
        #         A_plus = A[[x != s for x in A]];
        #         A_plus = vector_to_matrix(A_plus);
        #         Φ2 = evaluate(A_plus,x,mop);
        #         Φ2 = transpose(Φ2);
        #         a_tmp, ymod = regress(Φ2,y);
        #         r2_new = 0.0;
        #         r2 = 1 - lack_of_fit(y,ymod);
        #         A_plus = matrix_to_vector(A_plus);
        #         if abs(R2 - r2) < ε_backward || r2 == 1.0
        #             A = A_plus;
        #         else
        #             push!(A_plus,Vector(s))
        #         end
        #     end
        #     A = A_plus
        # end
        # A = vector_to_matrix(A);
        A = backwardstep(numu,p,A,x,y,R2,ε_backward,mop)


        ##### CHECK TERMINATION CRITERION
        if size(A)[1]  != 0
            Φ3 = evaluate(A,x,mop)
            Φ3 = transpose(Φ3)
            a_tmp, ymd = regress(Φ3,y)
            R2 = 1 - lack_of_fit(y,ymd)
            A = matrix_to_vector(A);
            if R2 >= accuracy || isapprox(R2,accuracy,atol=1e-15)
                temp_ind = findall(x->x>1e-4,a_tmp)
                A = A[temp_ind]
                a_tmp = a_tmp[temp_ind]
                println("R2: ", R2)
                println("coeff: ", a_tmp)
                println("indices: ", A)
                println("Accuracy achieved. Breaking\n")
                return a_tmp,A
            end
        end
        ######
    end
    error("Algorithm terminated early; perhaps a pathological problem was provided.")
end
