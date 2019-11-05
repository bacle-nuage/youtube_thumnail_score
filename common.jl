
```
model
```
mutable struct Model:
    input
    layours
    output
    error
end

```
layour
```
struct Layour:
    func
    back_func
end

'''
model作成
'''
function create_model(d=None):
    input_ini = {'d':d}
    layour_ini = ([],[],[])
    model = Model(input_ini, layour_ini)
    return model
end

'''
層の追加
'''
function add_layour(model, func, back_func, d=None)
    push!(model.layours,Layour(func, back_func, d)
    return model
end

'''
出力層の追加
'''
function set_output(model, func)
    model.output = Layour(func,NULL)
    return model
end

'''
出力層の追加
'''
function set_error(model, func)
    model.error = Layour(func,NULL)
    return model
end

'''
学習
'''
function learn(model, x_train, t_train, x_test=None, t_test=None, batch_size=100, epoch=50, init_model_flag=True,
          optimizer=None, init_optimizer_flag=True, shuffle_flag=True, learn_info=None):
    if init_model_flag:
        model = init_model(model)
    if optimizer is None:
        optimizer = create_optimizer(SGD)
    if init_optimizer_flag:
        optimizer = init_optimizer(optimizer, model)

    # 学習情報初期化
    learn_info = epoch_hook(learn_info, epoch, 0, model, x_train, None, t_train, x_test, t_test, batch_size)

    # エポック実行
    for i in range(epoch):
        idx = np.arange(x_train.shape[0])
        if shuffle_flag:
            # データのシャッフル
            np.random.shuffle(idx)

        # 学習
        y_train = np.zeros(t_train.shape)
        for j in range(0, x_train.shape[0], batch_size):
            # propagation
            y_train[idx[j:j+batch_size]], err, us = propagation(model, x_train[idx[j:j+batch_size]], t_train[idx[j:j+batch_size]])
            # back_propagation
            dz, dus = back_propagation(model, x_train[idx[j:j+batch_size]], t_train[idx[j:j+batch_size]], y_train[idx[j:j+batch_size]], us)
            # update_weight
            model, optimizer = update_weight(model, dus, optimizer)

        # 学習情報設定(エポックフック)
        learn_info = epoch_hook(learn_info, epoch, i+1, model, x_train, y_train, t_train, x_test, t_test, batch_size)

    return model, optimizer, learn_info
end

```
予測
```
function predict(model, x_pred, t_pred=None):
    y_pred, err, us = propagation(model, x_pred, t_pred, learn_flag=False)
    return y_pred, err, us
end

'''
正解率計算
'''
function accuracy_rate(y, t):
    # 2値分類
    if t.shape[t.ndim-1] == 1:
        round_y = np.round(y)
        return np.sum(round_y == t)/y.shape[0]
    # 多クラス分類
    else:
        max_y = np.argmax(y, axis=1)
        max_t = np.argmax(t, axis=1)
        return np.sum(max_y == max_t)/y.shape[0]
end

'''
保存
'''
function save_model(model, file_path):
    f = open(file_path, "wb")
    pickle.dump(model, f)
    f.close
end

'''
復元
'''
function load_model(file_path):
    f = open(file_path, "rb")
    model = pickle.load(f)
    f.close
    return model
end

function propagation(model, x, t=None, learn_flag=True):
    us = {}
    u = x
    err = None
    weight_decay_sum = 0
    # layer
    for k, v in model["layer"].items():
        # propagation関数設定
        propagation_func = middle_propagation
        if v["func"].__name__ + "_propagation" in globals():
            propagation_func = eval(v["func"].__name__ + "_propagation")
        # propagation関数実行
        us[k], weight_decay_r = propagation_func(v["func"], u, v["weights"], model["weight_decay"], learn_flag, **v["params"])
        u = us[k]["z"]
        weight_decay_sum = weight_decay_sum + weight_decay_r
    # output
    if "output" in model:
        propagation_func = output_propagation
        # propagation関数実行
        us["output"] = propagation_func(model["output"]["func"], u, learn_flag, **model["output"]["params"])
        u = us["output"]["z"]
    # error
    y = u
    # 学習時には、誤差は計算しない
    if learn_flag == False:
        if "error" in model:
            if t is not None:
                err = model["error"]["func"](y, t)
        # 重み減衰
        if "weight_decay" is not None:
            if learn_flag:
                err = err + weight_decay_sum

    return y, err, us
end

function back_propagation(model, x=None, t=None, y=None, us=None, du=None):
    dus = {}
    if du is None:
        # 出力層+誤差勾配関数
        output_error_back_func = eval(model["output"]["func"].__name__ + "_" + model["error"]["func"].__name__ + "_back")
        du = output_error_back_func(y, us["output"]["u"], t)
        dus["output"] = {"du":du}
    dz = du
    for k, v in reversed(model["layer"].items()):
        # back propagation関数設定
        back_propagation_func = middle_back_propagation
        if v["func"].__name__ + "_back_propagation" in globals():
            back_propagation_func = eval(v["func"].__name__ + "_back_propagation")
        # back propagation関数実行
        dus[k] = back_propagation_func(v["back_func"], dz, us[k], v["weights"], model["weight_decay"], v["calc_du_flag"], **v["params"])
        dz = dus[k]["du"]
        # du計算フラグがFalseだと以降計算しない
        if v["calc_du_flag"] == False:
            break

    return dz, dus
end

function update_weight(model, dus, optimizer):
    for k, v in model["layer"].items():
        # 重み更新
        if v["func"].__name__ + "_update_weight" in globals():
            update_weight_func = eval(v["func"].__name__ + "_update_weight")
            v["weights"], optimizer["stats"][k] = update_weight_func(optimizer["func"], dus[k], v["weights"], optimizer["stats"][k], **optimizer["params"])
    return model, optimizer
end

function middle_propagation(func, u, weights, weight_decay, learn_flag, **params):
    z = func(u, **params)
    return {"u":u, "z":z}, 0
end

function middle_back_propagation(back_func, dz, us, weights, weight_decay, calc_du_flag, **params):
    du = back_func(dz, us["u"], us["z"], **params)
    return {"du":du}
end

function output_propagation(func, u, learn_flag, **params):
    z = func(u, **params)
    return {"u":u, "z":z}
end
