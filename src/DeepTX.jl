module DeepTX

using Catalyst
using Random
using BlackBoxOptim
using Flux
using CSV
using DataFrames
using JLD2
using Sobol
using Statistics
include("train_NN.jl")
include("utils_GTM.jl")
include("constants.jl")

d=5
function loss(param, hist_yobs,model)
    pred = pred_pdf_infe(model, param, 0:length(hist_yobs))
    # println("the params is $param")
    # println("the pred is $pred")
    # println("the hist_yobs is $hist_yobs")
    loss =  hellinger2(pred, hist_yobs)
    # print("the loss is $loss")
    return hellinger2(pred, hist_yobs)
end

function TX_inferrer(hist_yobs,model,param=[],  η=1.1, 
    max_epoch=50, patience=5, min_delta=1e-4,sample_id=1)
    logranges = [
        1.0 15.0;
        0.1 10.0;
        1.0 15.0;
        0.01 10.0;
        0.1 400.0
    ]
    
    best_loss = Inf
    wait = 0
    temp_loss = Inf
    if length(param)==0
        prior = Product(Uniform.(logranges[1:d, 1], logranges[1:d, 2]))
        param = rand(prior)
    end
    for epoch in 1:max_epoch
        grads = gradient(p -> loss(p, hist_yobs, model), param)
        # print("the grads values is $grads")
        grad_values = grads[1]
        # grad_values .= map(x -> x > 0 ? clamp(x, 0.005, 10.0) : clamp(x, -10.0, -0.005), grad_values)

        new_param = param .- η .* grad_values
        for i in 1:length(param)
            new_param[i] = clamp(new_param[i], logranges[i,1], logranges[i,2])
        end
        param .= new_param
        pred = pred_pdf_infe(model, param, 0:length(hist_yobs))
        temp_loss = hellinger2(pred, hist_yobs)
        # println(grads)
        # println("Sample $sample_id | Epoch $epoch | Loss = $temp_loss (Thread $(Threads.threadid()))")

        # Early stopping
        # if temp_loss + min_delta < best_loss
        #     best_loss = temp_loss
        #     wait = 0
        # else
        #     wait += 1
        #     if wait >= patience
        #         println("Sample $sample_id | Early stopping triggered at epoch $epoch with best loss $best_loss")
        #         break
        #     end
        # end
    end
    pred = pred_pdf_infe(model, param, 0:length(hist_yobs))
    temp_loss = hellinger2(pred, hist_yobs)
    println("Sample $sample_id |  Loss = $temp_loss (Thread $(Threads.threadid()))")
    return param, temp_loss
end

function TX_inferrers(
    hist_yobs_list; 
    η_list=nothing, 
    epoch_list=nothing, 
    patience=5, 
    min_delta=1e-4)
    
    @load joinpath(@__DIR__, "..", "assets", "model_stats_prob.jld2") model
    n = length(hist_yobs_list)
    params_list = Vector{Vector{Float64}}(undef, n)
    temp_loss_list = Vector{Float64}(undef, n)

    
    if η_list === nothing
        η_list = fill(1, n)
    end
    if epoch_list === nothing
        epoch_list = fill(40, n)
    end
    param = []
    # Threads.@threads for j in 1:n
    for j in 1:n

        hist_yobs = hist_yobs_list[j]
        η = η_list[j]
        max_epoch = epoch_list[j]
        # param, temp_loss = TX_inferrer!(
        #     hist_yobs,param,  η, max_epoch, patience, min_delta, j
        # )

        max_trials = 15 
        trial = 0

        trial_params = []
        trial_losses = Float64[]
      
        while true
            trial += 1
            param = []
            param, temp_loss = TX_inferrer(
                hist_yobs,model, param, η, max_epoch, patience, min_delta, j
            )
            
            println("Sample $j | Trial $trial | Temp Loss = $temp_loss")
        
           
            push!(trial_params, copy(param))
            push!(trial_losses, temp_loss)
        
            if temp_loss <= 0.1
                println("Sample $j | Stopping condition reached early (Temp Loss = $temp_loss, Trial = $trial)")
                params_list[j] = copy(param)
                temp_loss_list[j] = temp_loss
                break
            elseif trial >= max_trials
                trial_losses = replace(trial_losses, NaN => 10)
                best_idx = argmin(trial_losses) 
                params_list[j] = copy(trial_params[best_idx])
                temp_loss_list[j] = trial_losses[best_idx]
                println("Sample $j | Max trials reached, picking best trial $best_idx with loss $(trial_losses[best_idx])")
                break
            end
        end

    end

    return params_list, temp_loss_list
end



function loss_hellinger_map(x::AbstractVector, model,hist_yobs, tt=tt)
    bufs = zeros(Threads.nthreads())
    Threads.@threads for i in 1:length(tt)
        ps = x
        # pred = pred_pdf(model, ps, 0:length(hist_yobs))

        pred = pred_pdf_infe(model, ps, 0:length(hist_yobs))
        bufs[Threads.threadid()] += hellinger2(pred, hist_yobs)
    end
    sum(bufs)
end

# infer the param value of the test set by neural network.

function inference_parameters(gene_exp_data,init_params=[])
    include("nnet.jl")

    @load joinpath(@__DIR__, "..", "assets", "model_stats_prob.jld2") model
    model = model.nn
    d = 5
    intensity = 1
    logranges = [  1.0 15.0
                    0.1 10.0
                    1.0 15.0
                    0.01 10.0
                    0.1  400.0
                ]

    op4 = names(gene_exp_data)
    print(length(op4))
    estimates = Vector{Float64}[]
    for j in eachindex(op4)
        gene_exp = gene_exp_data[:, j]
        hist_yobs = convertCountsToProb(gene_exp)
        tt = [0, 0]
        yobs = []
        if length(init_params)==0
            prior = Product(Uniform.(logranges[1:d, 1], logranges[1:d, 2]))
            init_param = rand(prior)
        else
            init_param=init_params[j]
        end

        Threads.@threads for i in [1]
            opt_result = bboptimize(
                p -> loss_hellinger_map(p, model, hist_yobs, tt),init_param;
                SearchRange = [tuple(logranges[i, :]...) for i = 1:d],
                TraceMode = :silent,
            )
            push!(estimates, best_candidate(opt_result))
            println(best_candidate(opt_result))
        end
    end
    estimates
end

function calculate_bs_bf(estimates)
    burst_freq = 1 ./ (estimates[:,1] ./ estimates[:,2] .+ estimates[:,3] ./ estimates[:,4])
    burst_size = estimates[:,5] .* (estimates[:,1] ./ estimates[:,2])
    mean_val = burst_freq.*burst_size
    burst_freq,burst_size,mean_val
end

function save_csv(estimates,file_path,gene_name_arr)
    estimates_df = DataFrame(estimates, :auto)
    estimates_df = DataFrame(Matrix(estimates_df)', :auto);
    estimates_df.gene_name = gene_name_arr
    CSV.write(file_path, estimates_df)
end

function matrix_to_dataframe(estimates,gene_name_arr)
    estimates_df = DataFrame(estimates, :auto)
    estimates_df = DataFrame(Matrix(estimates_df)', :auto);
    estimates_df.gene_name = gene_name_arr
    estimates_df
end

function calculate_stats(gene_exp_data,estimated_params)
    mean_idu = mean.(eachcol(gene_exp_data))
    var_idu = var.(eachcol(gene_exp_data))
    burst_freq_idu,burst_size_idu,mean_es_idu = calculate_bs_bf(estimated_params)

    estimated_params.bf=burst_freq_idu
    estimated_params.bs=burst_size_idu
    estimated_params.mean_es = mean_es_idu
    estimated_params.mean_true = mean_idu
    estimated_params.var_true = var_idu
    estimated_params
end 


export inference_parameters,Split,MNBModel,InputLayer,convertCountsToProb,TX_inferrers,calculate_stats,matrix_to_dataframe
end # module
