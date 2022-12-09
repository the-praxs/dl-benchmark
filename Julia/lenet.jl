module lenet
export LeNet

using Flux
using Flux: flatten

function LeNet(; imgsize=(28, 28, 1), nclasses=10)
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)

    return Chain(
    # First convolution, operating upon 28x28 image
    Conv((5, 5), imgsize[end]=>6, relu),
    MaxPool((2, 2)),

    # second convolution, operating upon 14x14 image
    Conv((5, 5), 6=>16, relu),
    MaxPool((2, 2)),

    # Third convolution, operating upon 7x7 image
    #Conv((3,3), 32=>32, pad=(1,1), relu),
    #MaxPool((2,2)),

    Flux.flatten,
    Dense(prod(out_conv_size), 120, relu),
    Dense(120, 84, relu),
    Dense(84, nclasses)
    )
end

end