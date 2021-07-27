from confidnet.loaders import usualtorch_loader as dload
from confidnet.utils.misc import load_yaml

def get_loader(config_args):
    """
        Return a new instance of dataset loader
    """

    # Available models
    data_loader_factory = {
        "mdrdc": dload.MDRDCLoader
    }
    print('Data loading ...')

    return data_loader_factory[config_args['data']['dataset']](config_args=config_args)

if __name__=="__main__":
    config_path='/.../confidnet/confs/exp_text_server.yaml'
    config_args= load_yaml(config_path)
    out=get_loader(config_args)
    print(out)
