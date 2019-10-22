function prop(u, pool_size=2, padding=0, strides=None):
    # ストライド設定
    if strides is None:
        strides = pool_size
    # プールサイズ
    if type(pool_size) == int:
        pool_size_h, pool_size_w = pool_size, pool_size
    else:
        pool_size_h, pool_size_w = pool_size
    # パディング
    if type(padding) == int:
        padding_h, padding_w = padding, padding
    else:
        padding_h, padding_w = padding
    # ストライド
    if type(strides) == int:
        strides_h, strides_w = strides, strides
    else:
        strides_h, strides_w = strides
    # 画像の数、高さ、幅、チャネル
    un, uh, uw, uc = shape(u)
    # パディング後の高さ、幅
    ph, pw = uh + padding_h*2, uw + padding_w*2
    # パディング
    pu = zeros(un, ph, pw, uc)
    pu[:, padding_h:padding_h+uh, padding_w:padding_w+uw, :] = u
    # プーリング後の高さ、幅
    zh, zw = int((ph-pool_size_h)/strides_h)+1, int((pw-pool_size_w)/strides_w)+1
    # プーリング後の格納領域確保
    z = zeros(un, zh, zw, uc)
    # プーリング後の高さ、幅分ループ
    for i in range(zh):
        for j in range(zw):
            # 画像からプーリングサイズ分切り出し
            # pus - (un, pool_size_h, pool_size_w, uc)
            pus = pu[:, i*strides_h:i*strides_h+pool_size_h, j*strides_w:j*strides_w+pool_size_w, :]
            # 切り出した画像の最大
            z[:, i, j, :] = max(reshape(pus, un, pool_size_h*pool_size_w, uc), axis=1)
    return z
end

function back_prop(dz, u, z, pool_size=2, padding=0, strides=None):
    # ストライド設定
    if strides is None:
        strides = pool_size
    # プールサイズ
    if type(pool_size) == int:
        pool_size_h, pool_size_w = pool_size, pool_size
    else:
        pool_size_h, pool_size_w = pool_size
    # パディング
    if type(padding) == int:
        padding_h, padding_w = padding, padding
    else:
        padding_h, padding_w = padding
    # ストライド
    if type(strides) == int:
        strides_h, strides_w = strides, strides
    else:
        strides_h, strides_w = strides
    # 画像の数、高さ、幅、チャネル
    un, uh, uw, uc = shape(u)
    # パディング後の高さ、幅
    ph, pw = uh + padding_h*2, uw + padding_w*2
    # パディング
    pu = zeros(un, ph, pw, uc)
    pu[:, padding_h:padding_h+uh, padding_w:padding_w+uw, :] = u
    # プーリング後のデータ格納用
    zh, zw = int((ph-pool_size_h)/strides_h)+1, int((pw-pool_size_w)/strides_w)+1
    # 勾配格納領域確保
    dpu = zeros(un, ph, pw, uc)
    # プーリング後の高さ、幅分ループ
    for i in range(zh):
        for j in range(zw):
            # dzijの取り出し
            # dzs - (un, fn)
            dzs = dz[:,i,j,:]
            # 最大の要素
            pus = pu[:, i*strides_h:i*strides_h+pool_size_h, j*strides_w:j*strides_w+pool_size_w, :]
            # 順番並び替え
            pustr = pus.transpose(0, 3, 1, 2)
            # 最大の要素
            z_argmax = rgmax(reshape(pustr, un*uc, pool_size_h*pool_size_w), axis=1)
            # 勾配格納
            dudrtr = zeros(un*uc, pool_size_h*pool_size_w)
            dudrtr[arange(un*uc), z_argmax.ravel()] = dzs.ravel()
            dudtr = reshape(dudrtr, un, uc, pool_size_h, pool_size_w)
            # 順番元に戻す
            dud = dudtr.transpose(0, 2, 3, 1)
            # 勾配設定
            dpu[:, i*strides_h:i*strides_h+pool_size_h, j*strides_w:j*strides_w+pool_size_w, :] += dud
    du = dpu[:, padding_h:padding_h+uh, padding_w:padding_w+uw, :]
    return du
end
