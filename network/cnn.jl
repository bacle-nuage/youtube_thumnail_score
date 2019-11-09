
type TwoLayerNet{T}
    conv1lyr::ConvolutionLayer{T}
    relu1lyr::ReluLayer
    pool1lyr::PoolingLayer

    conv2lyr::ConvolutionLayer{T}
    relu2lyr::ReluLayer
    pool2lyr::PoolingLayer

    a1lyr::AffineLayer{T}
    relu3lyr::ReluLayer

    a2lyr::AffineLayer{T}

    squred_error::SquredErrorLossLayer{T}
end

function (::Type{TwoLayerNet{T}}){T}(input_size::Int, hidden_size::Int, hidden_size2::Int, hidden_size3::Int, output_size::Int, weight_init_std::Float64=0.01)
    # conv ini
    W1 = weight_init_std .* randn(T, hidden_size, input_size)
    b1 = zeros(T, hidden_size)
    # conv ini
    W2 = weight_init_std .* randn(T, hidden_size2, hidden_size)
    b2 = zeros(T, hidden_size2)
    # affine ini
    W3 = weight_init_std .* randn(T, hidden_size3, hidden_size2)
    b3 = zeros(T, hidden_size3)
    # affine ini
    W4 = weight_init_std .* randn(T, output_size, hidden_size3)
    b4 = zeros(T, output_size)

    #conv lyr
    conv1lyr = ConvolutionLayer(W1, b1)
    relu1lyr = ReluLayer()
    pool1lyr = PoolingLayer()
    #conv lyr
    conv2lyr = ConvolutionLayer(W1, b1)
    relu2lyr = ReluLayer()
    pool2lyr = PoolingLayer()
    # affine lyr
    a1lyr = AffineLayer(W1, b1)
    relu1lyr = ReluLayer()
    # affine lyr
    a2lyr = AffineLayer(W2, b2)
    # loss
    squred_error = SquredErrorLossLayer{T}()

    TwoLayerNet(conv1lyr, relu1lyr, pool1lyr, conv2lyr, relu2lyr, pool2lyr, a1lyr, relu1lyr, a2lyr, squred_error)
end

function predict{T}(net::TwoLayerNet{T}, x::AbstractArray{T})
    c1 = forward(net.conv1lyr, x)
    r1 = forward(net.relu1lyr, c1)
    p1 = forward(net.pool1lyr, r1)

    c2 = forward(net.conv2lyr, p1)
    r2 = forward(net.relu2lyr, c2)
    p2 = forward(net.pool2lyr, r2)

    a1 = forward(net.a1lyr, p2)
    r3 = forward(net.relu3lyr, a1)

    a2 = forward(net.a1lyr, r3)

    r4 = forward(net.relu1lyr, a2)
    # squred_error(r4)
    r4
end

function loss{T}(net::TwoLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T})
    y = predict(net, x)
    forward(net.squred_error, y, t)
end

function accuracy{T}(net::TwoLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T})
    y = vec(mapslices(indmax, predict(net, x), 1))
    mean(y .== t)
end

immutable TwoLayerNetGrads{T}
    W1::AbstractMatrix{T}
    b1::AbstractVector{T}
    W2::AbstractMatrix{T}
    b2::AbstractVector{T}
    W3::AbstractMatrix{T}
    b3::AbstractVector{T}
    W4::AbstractMatrix{T}
    b4::AbstractVector{T}
end

function Base.gradient{T}(net::TwoLayerNet{T}, x::AbstractArray{T}, t::AbstractArray{T})
    # forward
    loss(net, x, t)
    # backward
    dout = one(T)
    dr4 = backward(net.squred_error, dout)

    da2 = backward(net.a2lyr, dr4)

    dr3 = backward(net.relu3lyr, dr3)
    da1 = backward(net.a1lyr, dr3)

    dp2 = backward(net.pool2lyr, dz2)
    dr2 = backward(net.relu2lyr, da2)
    dc2 = backward(net.conv2lyr, dz1)

    dp1 = backward(net.pool1lyr, dz2)
    dr1 = backward(net.relu1lyr, da2)
    dc1 = backward(net.conv1lyr, dz1)

    TwoLayerNetGrads(net.conv1lyr.dW, net.conv1lyr.db, net.conv2lyr.dW, net.conv2lyr.db, net.a1lyr.dW, net.a1lyr.db, net.a2lyr.dW, net.a2lyr.db)
end

function applygradient!{T}(net::TwoLayerNet{T}, grads::TwoLayerNetGrads{T}, learning_rate::T)
    net.conv1lyr.W -= learning_rate .* grads.W1
    net.conv1lyr.b -= learning_rate .* grads.b1
    net.conv2lyr.W -= learning_rate .* grads.W2
    net.conv2lyr.b -= learning_rate .* grads.b2

    net.a1lyr.W -= learning_rate .* grads.W3
    net.a1lyr.b -= learning_rate .* grads.b3
    net.a2lyr.W -= learning_rate .* grads.W4
    net.a2lyr.b -= learning_rate .* grads.b4
end

function get_data(path = None)
    _x_train, _t_train, _x_test, _t_test
end

# _x_train, _t_train = traindata();
# _x_test, _t_test = testdata();
_x_train, _t_train, _x_test, _t_test = get_data()
x_train = collect(Float32, _x_train) ./ 255   # 型変換+正規化
t_train = onehot(Float32, _t_train, 0:9)      # One-hot Vector 化
x_test = collect(Float32, _x_test) ./ 255     # 型変換+正規化
t_test = onehot(Float32, _t_test, 0:9)        # One-hot Vector 化

iters_num = 10000;
train_size = size(x_train, 2); # => 60000
batch_size = 100;
learning_rate = Float32(0.1);
train_loss_list = Float32[];
train_acc_list = Float32[];
test_acc_list = Float32[];

iter_per_epoch = max(train_size ÷ batch_size, 1)  # => 600

network = TwoLayerNet{Float32}(784, 50, 10);
for i = 1:iters_num
    batch_mask = rand(1:train_size, batch_size)
    x_batch = x_train[:, batch_mask]
    t_batch = t_train[:, batch_mask]

    # 誤差逆伝播法によって勾配を求める
    grads = gradient(network, x_batch, t_batch)

    # 更新
    applygradient!(network, grads, learning_rate)

    _loss = loss(network, x_batch, t_batch)
    push!(train_loss_list, _loss)

    if i % iter_per_epoch == 1
        train_acc = accuracy(network, x_train, t_train)
        test_acc = accuracy(network, x_test, t_test)
        push!(train_acc_list, train_acc)
        push!(test_acc_list, test_acc)
        println("$(i-1): train_acc=$(train_acc) / test_acc=$(test_acc)")
    end
end

final_train_acc = accuracy(network, x_train, t_train)
final_test_acc = accuracy(network, x_test, t_test)
push!(train_acc_list, final_train_acc)
push!(test_acc_list, final_test_acc)
println("final: train_acc=$(final_train_acc) / test_acc=$(final_test_acc)")
