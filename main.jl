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
function convolution(img, param)
  conv_img = imfilter(img, kernel.shape(param[1],param[2]))
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
function main(imgs,label,w,b,i,img_size)
  # input
  # グレースケールに変換
  g_img = color_to_gray(imgs[i])
  # リサイズ　TODO padarrayにすべき？
  img = imresize(g_img,[img_size,img_size]

  # Convolution
  # conv_img = convolution(img, [7,8])　TODO フィルタのかけ方調査の必要あり
  w1 = w
  conv_img = convolution(img, w)1

  # ReLU
  relu = ReLU(conv_img)

  # Convolution_2
  w2 = w
  conv_img2 = convolution(conv_img, w2)

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

# エポック数
epoch  = 100
# 出力の数
output = 100
# 画像サイズ
img_size = 299
# 色 RGBのため３
color = 1
# 係数
param = 0.1
# 重み
rng = MersenneTwister(1234);
weghit = param * randn(rng, ComplexF32, (img_size, img_size) * color)
# バイアス　個性
bias = zeros()

for i in images.length
  result = main(images,labels,weghit,bias,i,img_size)
end

#= installするパッケージ
Pkg.add("ImageFilterings")
Pkg.add("CSV")
Pkg.add("AffineTransforms")
Pkg.add("Random")　デフォルトで入っている？
Pkg.add("Images")
=#
