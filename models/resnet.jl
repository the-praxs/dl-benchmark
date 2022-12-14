module resnet
export ResNet

using Flux
using Flux: @functor

struct ResidualBlock
    conv_layers
    norm_layers
    shortcut
end

@functor ResidualBlock

# ResidualBlock Function allows us to define a Residual Block having any number of Convolution and Batch Normalization Layers
function ResidualBlock(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, strides::Array{Tuple{Int,Int}}, shortcut = identity)
    local conv_layers = []
    local norm_layers = []
    for i in 2:length(filters)
        push!(conv_layers, Conv(kernels[i-1], filters[i-1]=>filters[i], pad = pads[i-1], stride = strides[i-1]))
        push!(norm_layers, BatchNorm(filters[i]))
    end
    ResidualBlock(Tuple(conv_layers),Tuple(norm_layers),shortcut)
end

# Function converts the Array of scalar kernel, pad and stride values to tuples
function ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut = identity)
    ResidualBlock(filters, [(i,i) for i in kernels], [(i,i) for i in pads], [(i,i) for i in strides], shortcut)
end

function (block::ResidualBlock)(input)
    local value = copy.(input)
    for i in 1:length(block.conv_layers)-1
        value = relu.((block.norm_layers[i])((block.conv_layers[i])(value)))
    end
    relu.(((block.norm_layers[end])((block.conv_layers[end])(value))) + block.shortcut(input))
end

# Function to generate the residual blocks for ResNet18 and ResNet34
function BasicBlock(filters::Int, downsample::Bool = false, res_top::Bool = false)
    if(!downsample || res_top)
        return ResidualBlock([filters for i in 1:3], [3,3], [1,1], [1,1])
    end
    shortcut = Chain(Conv((3,3), filters÷2=>filters, pad = (1,1), stride = (2,2)), BatchNorm(filters))
    ResidualBlock([filters÷2, filters, filters], [3,3], [1,1], [1,2], shortcut)
end

# Function to build Resnet-18 model as described in the paper "Deep Residual Learning for Image Recognition"
function ResNet(Block::Function = BasicBlock, layers::Array{Int} = [2, 2, 2, 2], initial_filters::Int = 64, nclasses::Int = 10)

    local top = []
    local residual = []
    local bottom = []

    push!(top, Conv((7,7), 3=>initial_filters, pad = (3,3), stride = (2,2)))
    push!(top, x -> maxpool(x, (3,3), pad = (1,1), stride = (2,2)))

    for i in 1:length(layers)
        push!(residual, Block(initial_filters, true, i==1))
        for j in 2:layers[i]
            push!(residual, Block(initial_filters))
        end
        initial_filters *= 2
    end

    push!(bottom, x -> meanpool(x, (7,7)))
    push!(bottom, x -> reshape(x, :, size(x,4)))
    push!(bottom, (Dense(2048, nclasses)))
    push!(bottom, softmax)

    Chain(top..., residual..., bottom...)
end

end