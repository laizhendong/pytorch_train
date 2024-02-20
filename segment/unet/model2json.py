
import os
import numpy as np
import torch
import json
from collections import defaultdict,OrderedDict
MODEL_PATH = "output/CP_final.pth"

if __name__ == "__main__":
    loaded = torch.load(MODEL_PATH)

    params_numpy = defaultdict(dict)
    params_numpy_short = OrderedDict()
    for key in loaded.keys():
        params_numpy[key] = {"shape": loaded[key].cpu().numpy().shape, "value": loaded[key].cpu().numpy().flatten().tolist()}
        params_numpy_short[key] = "{}".format(loaded[key].cpu().numpy().shape)



    json_content = json.dumps(params_numpy_short, indent=4)
    with open(os.path.splitext(MODEL_PATH)[0] + "_short.json", "w") as f:
        f.write(json_content)

    json_content = json.dumps(params_numpy)
    with open(os.path.splitext(MODEL_PATH)[0] + ".json", "w") as f:
        f.write(json_content)






