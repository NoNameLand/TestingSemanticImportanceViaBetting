

import sys
import numpy as np
from sklearn import metrics

root_dir = "../"
sys.path.append(root_dir)
import configs
import datasets
from ibydmt.utils.pcbm import PCBM
from ibydmt.utils.config import get_config
from ibydmt.utils.concept_data import get_dataset_with_concepts
from ibydmt.utils.data import get_dataset
from ibydmt.utils.result import TestingResults
from ibydmt.classifiers import ZeroShotClassifier
from ibydmt.tester import get_test_classes

config_name, concept_type = "awa2", "class"
skit_kw = {"testing.kernel_scale": 0.9, "testing.tau_max": 400}
cskit_kw = {"ckde.scale": 2000, "testing.kernel_scale": 0.9, "testing.tau_max": 800}

config = get_config(config_name)
backbone_configs = config.sweep(["data.backbone"])
dataset = get_dataset(config)
classes = dataset.classes

test_classes = get_test_classes(config)
test_classes_idx = [classes.index(c) for c in test_classes]