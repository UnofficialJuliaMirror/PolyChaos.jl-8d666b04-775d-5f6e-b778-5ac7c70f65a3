export orthosparse, orthosparseMulti

function regress(M,y)
  c = pinv(M)*y
  c, M*c
end

function lack_of_fit(y::Array{Float64,1},ymod::Array{Float64,1})
    sum(δy^2 for δy in y - ymod) / sum(δy^2 for δy in y .- mean(y))
end

function orthosparse(y::Array{Float64,1},x::Array{Float64,1},name::String,p_max::Int64;ε_forward::Float64,ε_backward::Float64,accuracy::Float64)
    p_index = 1:p_max
    op = OrthoPoly(name,p_max)
    y_bar = mean(y)
    A, A_plus = Array{Int64,1}(), Array{Int64,1}()
    push!(A,0); push!(A_plus,0)
    for i in p_index
        i >= p_max && break
        ############################# FORWARD STEP #############################
        push!(A,i)
        Φ1 = evaluate(vec(A),x,op)
        a_tmp, ymod = Array{Float64,1}(),Array{Float64,1}()
        a_hat, ymod = regress(Φ1,y)
        R2 = 1 - lack_of_fit(y,ymod)
        #R2 >= ε_forward ? push!(A_plus,i) : nothing
        if R2 >= ε_forward
            push!(A_plus,i)
        end
        #R2 >= ε_forward ? A_plus = filter(x -> x != i, A) : nothing
        ########################### Backward STEP ##############################
        if A_plus[end] == i
            #Φ = evaluate(filter(x -> x ≠ b, A_plus),x,op)
            Φ2 = evaluate(i,x,op)
            a_tmp, ymd = Array{Float64,1}(),Array{Float64,1}()
            a_tmp, ymd = regress(Φ2,y)
            r2::Float64 = 0.0
            r2 = 1 - lack_of_fit(y,ymd)
            if r2 <= ε_backward
                #A_plus = filter(x -> x != i, A)
                filter!(x -> x != i, A)
                #pop!(A_plus)
                filter!(x -> x ≠ i, A_plus)
            end
        end
        #A_plus = A_plus[R2_array .< ε_backward]
        #A =  Array{Int64,1}()
        #A = A_plus
        ####################### CHECK TERMINATION CRITERION ####################
        if length(A) != 0

          Φ3 = evaluate(vec(A),x,op)
          a_tmp, ymd = Array{Float64,1}(),Array{Float64,1}()
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
    A = vcat(A,zeros(Int64,1,numu))
    #A_plus = vcat(A,zeros(Int64,1,numu))
    #A_minus = vcat(A,zeros(Int64,1,numu))

    for i in p_index
        i >= p_max && break
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
