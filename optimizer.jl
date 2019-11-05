function create_optimizer(func, **kwargs):
    optimizer = {"func":func, "params":kwargs}
    return optimizer
end

function init_optimizer(optimizer, model):
    optimizer_stats = {}
    for k, v in model["layer"].items():
        # オプティマイザの初期化
        if v["func"].__name__ + "_init_optimizer" in globals():
            init_optimizer_func = eval(v["func"].__name__ + "_init_optimizer")
            optimizer_stats[k] = init_optimizer_func()
    optimizer["stats"] = optimizer_stats

    return optimizer
end

function save_optimizer(optimizer, file_path):
    f = open(file_path, "wb")
    pickle.dump(optimizer, f)
    f.close
    return
end

function load_optimizer(file_path):
    f = open(file_path, "rb")
    optimizer = pickle.load(f)
    f.close
    return optimizer
end
