# packages
using ImageFilterings
using CSV
using AffineTransforms
using Random

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
  img = reshape(imgs[i],[img_size,img_size]

  # Convolution
  conv_img = convolution(img, [7,8])

  # ReLU
  relu = ReLU(conv_img)

  # Convolution_2
  conv_img2 = convolution(conv_img, [3,3])

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

# main
# initial
csv_file = /var/www/training.csv
csv_data = CSV.read(csv_file, header=false)

# 画像 image
images = csv_data[1]
# 期待値 label
labels = csv_data[2]

# 画像サイズ
img_size = 299
# 係数
param = 0.1

rng = MersenneTwister(1234);
# 重み
weghit = param * randn(rng, ComplexF32, (img_size, img_size))
# バイアス　個性
bias =

for i in images.length
  result = main(images,labels,weghit,bias,i,img_size)
end

#= installするパッケージ
Pkg.add("ImageFilterings")
Pkg.add("CSV")
Pkg.add("AffineTransforms")
=#
