using Common

module Convolution:

    def forward(u, W, b, padding=0, strides=1):
        () -> begin:
            # パディング
            padding_h, padding_w = Common.ChangeToTupleIfInt(padding)
            # ストライド
            strides_h, strides_w = Common.ChangeToTupleIfInt(strides)
            # 画像の数、高さ、幅、チャネル
            un, uh, uw, uc = shape(u)
            # フィルタの数、高さ、幅、チャネル
            fn, fh, fw, fc = shape(W)
            # パディング後の高さ、幅
            ph, pw = uh + padding_h*2, uw + padding_w*2
            # パディング
            pu = zeros(un, ph, pw, uc)
            pu[:, padding_h:padding_h+uh, padding_w:padding_w+uw, :] = u
            # 畳み込み後の高さ、幅
            zh, zw = int((ph-fh)/strides_h)+1, int((pw-fw)/strides_w)+1
            # 畳み込み後の格納領域確保
            z = zeros(un, zh, zw, fn)
            # 畳み込み後の高さ、幅分ループ
            for i in range(zh):
                for j in range(zw):
                    # 画像からフィルタサイズ分切り出し
                    # ud - (un, fh, fw, uc)
                    ud = pu[:, i*strides_h:i*strides_h+fh, j*strides_w:j*strides_w+fw, :]
                    # 切り出した画像とフィルタの内積の計算
                    # (un, fh*fw*fc) @ (fh*fw*fx, fn) => (un, fn)
                    z[:, i, j, :] = dot(reshape(ud, un, fh*fw*fc), reshape(W, fn, fh*fw*fc).T) + b
            return z
        end
    end

    function back_forward(dz, u, W, b, padding=0, strides=1):
        () -> begin:
            # パディング
            padding_h, padding_w = Common.ChangeToTupleIfInt(padding)
            # ストライド
            strides_h, strides_w = Common.ChangeToTupleIfInt(strides)
            # 画像の数、高さ、幅、チャネル
            un, uh, uw, uc = shape(u)
            # フィルターの数、高さ、幅、チャネル
            fn, fh, fw, fc = shape(W)
            # パディング後の高さ、幅
            ph, pw = uh + padding_h*2, uw + padding_w*2
            # パディング
            pu = zeros(un, ph, pw, uc)
            pu[:, padding_h:padding_h+uh, padding_w:padding_w+uw, :] = u
            # 畳み込み後の高さ、幅
            zh, zw = int((ph-fh)/strides_h)+1, int((pw-fw)/strides_w)+1
            # 勾配格納領域確保
            dpu = zeros(un, ph, pw, uc)
            dW = zeros(fn, fh, fw, fc)
            # 畳み込み後の高さ、幅分ループ
            for i in range(zh):
                for j in range(zw):
                    # dzijの取り出し
                    # dzs - (un, fn)
                    dzs = dz[:,i,j,:]
                    # u方向の勾配計算
                    # dpusr - (un,fn) @ (fn, fh*fw*fc) => (un, fh*fw*fc)
                    dpusr = dot(dzs, reshape(W,fn, fh*fw*fc))
                    # dpus - (un, fh, fw, uc)
                    dpus = reshape(dpusr,un, fh, fw, uc)
                    dpu[:, i*strides_h:i*strides_h+fh, j*strides_w:j*strides_w+fw, :] += dpus
                    # W方向の勾配計算
                    # pus  - (un, fh*fw*uc)
                    pus = pu[:, i*strides_h:i*strides_h+fh, j*strides_w:j*strides_w+fw, :]
                    # pudr - (un, fh*fw*uc)
                    pusr = reshape(pus,un, fh*fw*uc)
                    # dWsr - (fn, un) @ (un, fh*fw*uc) => (fn, fh*fw*uc)
                    dWsr = dot(dzs.T, pusr)
                    # dWs - (fn, fh, fw, fc=uc)
                    dWs = reshape(dWsr, fn, fh, fw, fc)
                    dW = dW + dWs
            # パディング部分の除去
            du = dpu[:, padding_h:padding_h+uh, padding_w:padding_w+uw, :]
            # b方向の勾配計算
            db = sum(reshape(dz, un*zh*zw, fn), axis=0)
            return du, dW, db
        end
    end

end
