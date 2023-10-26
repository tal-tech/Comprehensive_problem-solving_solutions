import yaml
from easydict import EasyDict as edict
import sys
import os
sys.path.append('../')
yaml_path_cfg = os.path.dirname(__file__) + '/../version.yaml'
try:

    with open(yaml_path_cfg, "r", encoding="utf-8") as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
except Exception as e:
    with open(yaml_path_cfg, "r", encoding="utf-8") as f:
        cfg = edict(yaml.load(f))

