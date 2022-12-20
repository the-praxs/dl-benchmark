using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Statistics, Random
import MLDatasets
import CUDA

include("models/lenet.jl")
using .lenet: LeNet

function prepare_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_loader, test_loader
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function eval_model(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

## utility functions
num_params(model) = sum(length, Flux.params(model)) 
round4(x) = round(x, digits=4)

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 0.1                 # learning rate
    λ = 5e-4                # L2 regularizer param, implemented as weight decay
    batchsize = 128         # batch size
    epochs = 10             # number of epochs
    seed = 42               # set seed > 0 for reproducibility
    use_cuda = true         # if true use cuda (if available)
    infotime = 1 	        # report every `infotime` epochs
    checktime = 5           # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = false        # log training with tensorboard
    savepath = "runs/"      # results path
end

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    
    if use_cuda
        device = gpu
        @info "Training on GPU..."
    else
        device = cpu
        @info "Training on CPU..."
    end

    ## DATA
    @info "Data Loading Time..."
    @time train_loader, test_loader = prepare_data(args)
    @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    model = LeNet() |> device
    @info "LeNet model: $(num_params(model)) trainable params"    
    
    ps = Flux.params(model)  

    # opt = ADAM(args.η) 
    opt = Descent(args.η)
    if args.λ > 0 # add weight decay, equivalent to L2 regularization
        opt = Optimiser(WeightDecay(args.λ), opt)
    end
    
    ## LOGGING UTILITIES
    function report(epoch)
        train = eval_model(train_loader, model, device)
        test = eval_model(test_loader, model, device)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end
    
    ## TRAINING
    @info "Evaluating model..."
    report(0)
    @time begin
        for epoch in 1:args.epochs
            @time begin
                for (x, y) in train_loader
                    x, y = x |> device, y |> device
                    gs = Flux.gradient(ps) do
                            ŷ = model(x)
                            loss(ŷ, y)
                        end
                
                    Flux.Optimise.update!(opt, ps, gs)
                end
            end
        
            ## Printing and logging
            epoch % args.infotime == 0 && report(epoch)
        end
    end
end

@time train()