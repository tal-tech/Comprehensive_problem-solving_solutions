import yaml
from easydict import EasyDict as edict
import os
try:
    with open("./east_for_monkey/config.yml", "r", encoding="utf-8") as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
except:
    try:
        with open("./east_for_monkey/config.yml", "r", encoding="utf-8") as f:
            cfg = edict(yaml.load(f))
    except:
        with open("config.yml", "r", encoding="utf-8") as f:
            cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
