export orthosparse, orthosparseMulti

function regress(M,y)
  c = pinv(M)*y
  c, M*c
end

function lack_of_fit(y::Array{Float64,1},ymod::Array{Float64,1})
    lof = sum(δy^2 for δy in y - ymod) / sum(δy^2 for δy in y .- mean(y))
    lof > 1 ? 1 : lof
end

function orthosparse(y::Vector{Float64},x::Vector{Float64},name::String,p_max::Int64;ε_forward::Float64,ε_backward::Float64,accuracy::Float64)
    p_index = 1:p_max
    final_index = Int64[]
    op = OrthoPoly(name,p_max)
    y_bar = mean(y)
    A, A_plus = [0], [0]
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

        ################ CHECK TERMINATION CRITERION ###################
        if length(A) != 0
          Φ = evaluate(A,x,op)
          a_tmp, ymod = regress(Φ,y)
          @show R2 = 1 - lack_of_fit(y,ymod)
          print("\n")
            # if r2 >= accuracy || R2 >= accuracy
            if R2 >= accuracy
                println("Accuracy achieved. Breaking.\n")
                return a_tmp, A
            end
        end
        #################################################################
        end
    error("Algorithm terminated early; perhaps a pathological problem was provided.")
end

function orthosparseMulti(y::Array{Float64,1},x::Matrix{Float64},name::String,p_max::Int64,numu::Int64,ε_forward::Float64,ε_backward::Float64,accuracy::Float64)
    p_index = 1:p_max
    j_index = 1:numu
    ############### To generate Multivariate ortho Polynomials #################
    ops = OrthoPoly[]
    for j = 1:numu
        push!(ops,OrthoPoly("gaussian",p_max))
    end
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
        ########################### FORWARD STEP ###############################
        for j in 1:numu
            I_p = calculateMultiIndices_interaction(numu,p_max,j,i)

            for k in I_p
                k = k'
                A = vcat(A,k)
                #push!(A, vec(k))
                #A_temp = transpose(hcat(A...))
                #Φ1 = evaluate(A_temp,x,mop)
                Φ1 = evaluate(A,x,mop)
                Φ1 = transpose(Φ1)
                a_hat, ymod = regress(Φ1,y)
                R2 = 1 - lack_of_fit(y,ymod)
                if R2 >= ε_forward
                    #A_plus  = vcat(A_plus,k)
                    A_plus  = vcat(A_plus,k)
                    #push!(A_plus, vec(k))
                end
            end
        end

        ########################## Backward STEP ###############################
        R2_array = Float64[]
        #A_plus = A_plus[[sum(x) == i for x in eachrow(A_plus)],:]
        A_temp = Array{Int64}(undef,0,numu)
        A_temp = A_plus[[sum(x) == i for x in eachrow(A_plus)],:]
        A_plus = A_plus[[sum(x) < i for x in eachrow(A_plus)],:]
        #for row in eachrow(A_plus)
        for row in eachrow(A_temp)
            row = Vector(row)
            Φ2 = evaluate(row,x,mop)
            #Φ = transpose(Φ)
            a_tmp, ymod = regress(Φ2,y)
            r2 = 1 - lack_of_fit(y,ymod)
            push!(R2_array,r2)
        end
        #A_plus = A_plus[R2_array .< ε_backward,:]
        A_temp = A_temp[R2_array .< ε_backward,:]
        A_plus = vcat(A_plus,A_temp)
        A = vcat(A,A_plus)
        display(A)
        ##################### CHECK TERMINATION CRITERION ######################
        if size(A)[1]  != 0
            Φ3 = evaluate(A,x,mop)
            Φ3 = transpose(Φ3)
            a_tmp, ymd = regress(Φ3,y)
            r2_accuracy = 1 - lack_of_fit(y,ymd)
            if r2_accuracy >= accuracy
                println("Accuracy achieved. Breaking.\n")
                return A
            end
        end
        ########################################################################
    end
    error("Algorithm terminated early; perhaps a pathological problem was provided.")
end
