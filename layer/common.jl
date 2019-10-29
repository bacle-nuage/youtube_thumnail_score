
module Common:

    function ChangeToTupleIfInt(param):
        if type(param) == int:
            param, param
        else:
            param
    end

end
