
module Flatten:

    function prop(u):
        # 画像の数、高さ、幅、チャネル
        un, uh, uw, uc = shape(u)
        z = reshape(u, un, uh*uw*uc)
        return z
    end

    function back_prop(dz, u, z):
        du = reshape(dz, shape(u))
        return du
    end

end
