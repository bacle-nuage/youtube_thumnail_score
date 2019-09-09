# packages
using ImageFilterings
using CSV
using AffineTransforms

# filter
function convolution(img, param)
  conv_img = imfilter(img, kernel.shape(param[1],param[1]))
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
function main(img,lavel,w,b) # TODO : w b はどうやって渡すべきか
  # input

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
  affine = Affine(w,b,relu3)

  #SquaredError

end

# main
# initial
csv_file = /var/www/training.csv
csv_data = CSV.read(csv_file, header=false)

# 画像
x = csv_data[1]
# 期待値
y = csv_data[2]

main(x,y)

#= installするパッケージ
Pkg.add("ImageFilterings")
Pkg.add("CSV")
Pkg.add("AffineTransforms")
=#
