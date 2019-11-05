function affine_init_layer(d_prev, d, weight_init_func=he_normal, weight_init_params={}, bias_init_func=zeros_b, bias_init_params={}):
    W = weight_init_func(d_prev, d, **weight_init_params)
    b = bias_init_func(d, **bias_init_params)
    return d, {"W":W, "b":b}
end

function affine_init_optimizer():
    sW = {}
    sb = {}
    return {"sW":sW, "sb":sb}
end

function affine_propagation(func, u, weights, weight_decay, learn_flag, **params):
    z = func(u, weights["W"], weights["b"])
    # 重み減衰対応
    weight_decay_r = 0
    if weight_decay is not None:
        weight_decay_r = weight_decay["func"](weights["W"], **weight_decay["params"])
    return {"u":u, "z":z}, weight_decay_r
end

function affine_back_propagation(back_func, dz, us, weights, weight_decay, calc_du_flag, **params):
    du, dW, db = back_func(dz, us["u"], weights["W"], weights["b"], calc_du_flag)
    # 重み減衰対応
    if weight_decay is not None:
        dW = dW + weight_decay["back_func"](weights["W"], **weight_decay["params"])
    return {"du":du, "dW":dW, "db":db}
end

function affine_update_weight(func, du, weights, optimizer_stats, **params):
    weights["W"], optimizer_stats["sW"] = func(weights["W"], du["dW"], **params, **optimizer_stats["sW"])
    weights["b"], optimizer_stats["sb"] = func(weights["b"], du["db"], **params, **optimizer_stats["sb"])
    return weights, optimizer_stats
end
