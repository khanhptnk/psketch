from .demonstration import DemonstrationTeacher

def load(config):
    cls_name = config.teacher.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such teacher: {}".format(cls_name))
