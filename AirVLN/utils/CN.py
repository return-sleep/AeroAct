import yacs.config


class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config() # Create a new Config instance such as CN.DATASET = "my_dataset" CN.MODEL.ARCH = "ResNet50"

