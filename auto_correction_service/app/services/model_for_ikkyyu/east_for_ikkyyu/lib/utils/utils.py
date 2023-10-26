import yaml
from easydict import EasyDict as edict
import os
yaml_path = os.path.dirname(__file__) + '/../../config.yml'
try:
    yaml_path = os.path.dirname(__file__) + '/../../config.yml'
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
except:
    try:
        with open("./east_for_ikkyyu/config.yml", "r", encoding="utf-8") as f:
            cfg = edict(yaml.load(f))
    except:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
