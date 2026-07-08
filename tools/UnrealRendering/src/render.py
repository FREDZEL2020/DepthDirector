import unreal
import os, sys
project_root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(project_root))
from renderer import RENDERERS

import json
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            args = json.load(f)
    else:
        exit()
    renderer = RENDERERS[args['renderer']](args)
    renderer.apply_render()