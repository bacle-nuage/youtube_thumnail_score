
```
model
```
mutable struct Model:
    layours
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
function create_model():
    layour_ini = []
    model = Model(layour_ini)
    return model
end

'''
層の追加
'''
function add_layour(func, back_funk; model = NONE)
    model
end

'''
学習
'''
function fit(args)
    body
end

'''
順伝播
'''
function fore_loop(args)
    body
end

'''
逆伝播
'''
function back_loop(args)
    body
end

'''
正解率計算
'''
function check_accuracy(args)
    body
end

'''
保存
'''
function save(args)
    body
end

'''
復元
'''
function load(args)
    body
end
