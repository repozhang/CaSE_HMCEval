
from confidnet.learners.default_learner import DefaultLearner
from confidnet.learners.selfconfid_learner import SelfConfidLearner


def get_learner(config, config_args, train_loader, val_loader, test_loader, start_epoch, device):
    """
        Return a new instance of model
    """
    # Available models
    learners_factory = {
        "default": DefaultLearner,
        "selfconfid": SelfConfidLearner,
    }

    return learners_factory[config_args["training"]["learner"]](
        config=config,
        config_args=config_args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        start_epoch=start_epoch,
        device=device
    )
if __name__=='__main__':
    learners_factory = {
        "default": DefaultLearner,
        "selfconfid": SelfConfidLearner,
    }
    print(learners_factory['selfconfid'])