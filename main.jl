#= 共通関数 =#
include("./common.jl")
include("./layer/convolution.jl")
include("./layer/relu.jl")
include("./layer/pooling.jl")
include("./layer/affine.jl")
include("./layer/flatten.jl")
include("./layer/squred_error.jl")

model = create_model
model = add_layour(model, )



# packages
using ImageFilterings
using CSV
using AffineTransforms
using Random
using Images

# カラー画像をグレーに変換
function color_to_gray(img)
  return Gray.(img)
end

# filter
function convolution(img, filter)
  conv_img = imfilter(img, filter)
  return conv_img
end

function Affine(W,b,x)
  tfm = AffineTransform(W,b)
  y = tfm * x
  y = tformfwd(tfm, x)
  y = similar(x); tformfwd!(y, tfm, x)
  return y
end

# 活性化関数
function ReLU(x)
  relu = maximum(0,x)
  return relu
end

# 誤差計算　SquaredError
function squared_error(y,y_pred)
  sqrt_e = sqrt(mean(y-y_pred^2))
  return sqrt_e

# 学習処理
function main(x_train, y_train, img_size, weghit, bias)
  # input
  # グレースケールに変換
  g_img = color_to_gray(imgs[i])
  # リサイズ　TODO padarrayにすべき？
  img = imresize(g_img,[img_size,img_size]

  # Convolution
  # conv_img = convolution(img, [7,8])　TODO フィルタのかけ方調査の必要あり
  conv_img = convolution(img, w[1])

  # ReLU
  relu = ReLU(conv_img)

  # Convolution_2
  conv_img2 = convolution(conv_img, w[2])

  # ReLU_2
  relu2 = ReLU(conv_img2)

  # Affine
  affine = Affine(w,b,relu2)

  # ReLU_3
  relu3 = ReLU(affine)

  # Affine_2
  affine2 = Affine(w,b,relu3)

  #SquaredError
  result = squared_error(y, y_pred)

  return result
end

# initial
# データ読み込み
csv_file  = /var/www/training.csv
dataframe = CSV.read(csv_file, header=false)
@show dataframe

# データの分割
train = csv_data[1][1:120]
test  = csv_data[1][121:end]
x_train, y_train = train[1], train[2]
x_test, y_test   = test[1], test[2]

# Param
epoch    = 1000  # 繰り返し数
img_size = 299   # 統一する画像サイズ
param    = 0.1   # 係数
rng = MersenneTwister(1234);
weghit = param * randn(rng, ComplexF32, (img_size, img_size) * color) # 重み
bias = zeros()   # バイアス　個性

# 基本処理
for i in 1:epoch
  result = main(x_train, y_train, img_size, weghit, bias)
end

#= installするパッケージ
Pkg.add("ImageFilterings")
Pkg.add("CSV")
Pkg.add("AffineTransforms")
Pkg.add("Random")　デフォルトで入っている？
Pkg.add("Images")
=#
