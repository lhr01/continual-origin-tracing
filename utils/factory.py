def get_model(model_name, args):
    name = model_name.lower()
    if name=="glpd":
        from models.glpd_simplecil import Learner
        return Learner(args)
    else:
        assert 0
