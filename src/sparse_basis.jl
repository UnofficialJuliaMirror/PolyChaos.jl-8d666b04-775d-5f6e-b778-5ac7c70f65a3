export orthosparse, orthosparseMulti
using LinearAlgebra

function regress(M,y)
  c = pinv(M)*y
  c, M*c
end

function lack_of_fit(y::Array{Float64,1},ymod::Array{Float64,1})
    lof = sum(δy^2 for δy in y - ymod) / sum(δy^2 for δy in y .- mean(y))
    lof > 1 ? 1 : lof
end

<<<<<<< HEAD
function matrix_to_vector(a::Array{Int64,2})
    return copy.(eachrow(a))
end

function vector_to_matrix(b::Array{Array{Int64,1},1})
    return reduce(vcat, transpose.(b))
end

=======
>>>>>>> 108e081a38491171990cb4a098847f0431dfee3f
function orthosparse(y::Vector{Float64},x::Vector{Float64},name::String,p_max::Int64;ε_forward::Float64,ε_backward::Float64,accuracy::Float64)
    p_index = 1:p_max
    final_index = Int64[]
    op = OrthoPoly(name,p_max)
    y_bar = mean(y)
<<<<<<< HEAD
    A, A_plus = [0], []
=======
    A, A_plus = [0], [0]
>>>>>>> 108e081a38491171990cb4a098847f0431dfee3f
    Φ = evaluate(A,x,op)
    a_hat, ymod = regress(Φ,y)
    R2 = 1 - lack_of_fit(y,ymod)
    for i in p_index
        i >= p_max && break
        ######################## FORWARD STEP ###########################

        push!(A,i)
        Φ = evaluate(A,x,op)
        a_hat, ymod = regress(Φ,y)
        @show R2_new = 1 - lack_of_fit(y,ymod)
        # if R2_new is significantlz better, then push to the basis
        if abs(R2_new - R2) >= ε_forward
            R2 = R2_new
            # do nothing
<<<<<<< HEAD
        elseif R2_new ==1.0
            R2 = R2_new ## to check if R2 is already 1.0
        else
            filter!(x -> x != i, A)
        end
        @show A_plus = A
        display("End of Forward step")

        ######################### Backward STEP #########################

        if A_plus[end] == i
            display("bwd step")
            for b in A_plus[1:end-1]
                Φ = evaluate(filter(x -> x ≠ b, A_plus),x,op)
                a_tmp, ymod = regress(Φ,y)
                @show r2 = 1 - lack_of_fit(y,ymod)
                if abs(R2 - r2) < ε_backward
                    display("filter in bwd step")
                    filter!(x -> x ≠ b, A_plus)
                end
            end
            A = A_plus
            display(A)
            display("End of Backward Step")
        end
=======
        else
        filter!(x -> x != i, A)
        end
        @show A_plus = A
        display("End of Forward step")
        #R2_new >= R2 ? R2 = R2_new : nothing
        # R2 <= ε_forward ? filter!(x -> x != i, A) : nothing
        #A_plus = A
        ######################### Backward STEP #########################
        R2_array = Float64[]
        for b in A_plus[1:end-1]
            Φ = evaluate(filter(x -> x ≠ b, A_plus),x,op)
            a_tmp, ymod = regress(Φ,y)
            @show r2 = 1 - lack_of_fit(y,ymod)
            push!(R2_array,r2)
        end
        #Here remove those Bases which do not cause significant change in R2 in backward setup
        s::Int64 = 1
        for r2 in R2_array
            if abs(R2 - r2) < ε_backward
                filter!(x -> x ≠ A_plus[s], A_plus)
                s -= 1
            end
            s += 1
        end
        @show A = A_plus
        display("End of Backward Step")
>>>>>>> 108e081a38491171990cb4a098847f0431dfee3f

        ################ CHECK TERMINATION CRITERION ###################
        if length(A) != 0
          Φ = evaluate(A,x,op)
          a_tmp, ymod = regress(Φ,y)
<<<<<<< HEAD
          @show R2_accuracy = 1 - lack_of_fit(y,ymod)
          print("\n")
            if R2_accuracy >= accuracy
=======
          @show R2 = 1 - lack_of_fit(y,ymod)
          print("\n")
            # if r2 >= accuracy || R2 >= accuracy
            if R2 >= accuracy
>>>>>>> 108e081a38491171990cb4a098847f0431dfee3f
                println("Accuracy achieved. Breaking.\n")
                return a_tmp, A
            end
        end
        #################################################################
<<<<<<< HEAD
    end
=======
        end
>>>>>>> 108e081a38491171990cb4a098847f0431dfee3f
    error("Algorithm terminated early; perhaps a pathological problem was provided.")
end

function orthosparseMulti(y::Array{Float64,1},x::Matrix{Float64},name::Array{String,1},p_max::Int64,numu::Int64,ε_forward::Float64,ε_backward::Float64,accuracy::Float64)
    p_index = 1:p_max
    j_index = 1:numu

    ####### Generates Multivariate ortho Polynomials
    ops = OrthoPoly[]
    for j = 1:length(name)
        push!(ops,OrthoPoly(name[j],p_max))
    end
<<<<<<< HEAD
    mop = MultiOrthoPoly(ops,p_max) # Multivariate Orthogonal Polynomials

    #####
    A = Array{Array{Int64,1},1}();
    A_plus = Array{Array{Int64,1},1}();
    push!(A,vec(zeros(Int64,1,numu)));
    A = vector_to_matrix(A);
    R2 = 0.0;
    #####

    Φ0 = evaluate(A,x,mop)
    Φ0 = transpose(Φ0)
    a_tmp,ymod = regress(Φ0,y)
    @show R2 = 1 - lack_of_fit(y,ymod)
    A = matrix_to_vector(A);
    for p in p_index
        p >= p_max && break

=======
    mop = MultiOrthoPoly(ops,p_max)
    ############################################################################
    A = Array{Int64}(undef, 0, numu)
    A_plus = Array{Int64}(undef, 0, numu)
    #A_minus = Array{Int64}(undef, 0, numu)
    #A = Vector{Vector{Int64}}()
    #A_plus = Vector{Vector{Int64}}()
    #push!(A, vec(zeros(1,numu)))
    #push!(A_plus, vec(zeros(1,numu)))
    #display(A)
    A = vcat(A,zeros(Int64,1,numu))
    #A_plus = vcat(A,zeros(Int64,1,numu))
    #A_minus = vcat(A,zeros(Int64,1,numu))
    for i in p_index
        i >= p_max && break
        #print("\n ##### \n")
        #display(A)
        #print("\n ##### \n")
        #if size(A)[1] == 0
        #    A = vcat(A,zeros(Int64,1,numu))
        #end
>>>>>>> 108e081a38491171990cb4a098847f0431dfee3f
        ########################### FORWARD STEP ###############################

        ##### Generating the set multiindecies of pth order
        I_p = Array{Array{Int64,1},1}();
        for j in j_index
            indices_set = calculateMultiIndices_interaction(numu,p_max,j,p)
            for k in indices_set
                push!(I_p,Vector(k))
            end
        end
        display("I_p set")
        display(I_p)
        println("\n")
        ############

        for j in I_p
            push!(A,Vector(j))
            A = vector_to_matrix(A);
            #display(A)
            Φ1 = evaluate(A,x,mop)
            Φ1 = transpose(Φ1)
            a_hat, ymod = regress(Φ1,y)
            @show R2_new = 1 - lack_of_fit(y,ymod)
            A = matrix_to_vector(A);

            # if R2_new is significantlz better, then push to the basis
            if R2_new == 1.0
                R2 = R2_new
                # do nothing
            elseif abs(R2_new - R2) > ε_forward
                R2 = R2_new
                # do nothing
            else
                display("filter in fwd step")
                pop!(A)
            end
        end
        # A_plus = A;
        display(A)
        display("End of Forward step")
        println("\n")
        ########################## Backward STEP ###############################

        @show A_a = A[[sum(x) == p for x in A]]; #set of pth order basis
        @show A_b = A[[sum(x) < p for x in A]]; #set of all basis of order less than p
        if A_a != []
            for s in A_b
                A_plus = A[[x != s for x in A]];
                @show A_plus = vector_to_matrix(A_plus);
                Φ2 = evaluate(A_plus,x,mop);
                Φ2 = transpose(Φ2);
                a_tmp, ymod = regress(Φ2,y);
                r2_new = 0.0;
                @show r2 = 1 - lack_of_fit(y,ymod);
                A_plus = matrix_to_vector(A_plus);
                if abs(R2 - r2) < ε_backward || r2 == 1.0
                    # display("filter in bwd step")
                    A = A_plus;
                else
                    push!(A_plus,Vector(s))
                end
            end
            A = A_plus
        end
        A = vector_to_matrix(A);
        display("A")
        display(A)
        display("End of Backward step")
        println("\n")

        ##################### CHECK TERMINATION CRITERION ######################

        if size(A)[1]  != 0
            display("check termination")
            Φ3 = evaluate(A,x,mop)
            Φ3 = transpose(Φ3)
            a_tmp, ymd = regress(Φ3,y)
            @show r2_accuracy = 1 - lack_of_fit(y,ymd)
            println("\n")
            A = matrix_to_vector(A);
            if r2_accuracy >= accuracy
                println("Accuracy achieved. Breaking.\n")
                return a_tmp,A
            end
        end
        ########################################################################
    end
    error("Algorithm terminated early; perhaps a pathological problem was provided.")
end
