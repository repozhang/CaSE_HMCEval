from confidnet.models.origin import ORIGIN
from confidnet.models.origin_selfconfid import ORIGINSelfConfid
from confidnet.models.origin_extractor import ORIGINExtractor


def get_model(config,config_args):
    """
        Return a new instance of model
    """

    # Available models
    model_factory = {
        "origin": ORIGIN,
        "origin_selfconfid": ORIGINSelfConfid,
        "origin_extractor": ORIGINExtractor
    }

    return model_factory[config_args["model"]["name"]]
