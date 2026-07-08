from .base import Storage
import os
import subprocess

class LocalStorage(Storage):
    def __init__(self, args, project_path):
        self.args = args
        self.PROJECT_PATH = project_path
        self.SRC_PROJECT_PATH = args.get("SRC_PROJECT_PATH")
        self.BAKED_PROJECT_PATH = args.get("BAKED_PROJECT_PATH")
        self.ASSETS_PATH = args.get("ASSETS_PATH")
        self.AVATAR_ASSETS_PATH = args.get("AVATAR_ASSETS_PATH",None)
    def check_project_exists(self, scene_name):
        return os.path.exists(f"{self.BAKED_PROJECT_PATH}/{scene_name}")
    
    def load_project(self, scene_name):
        # os.system(f"rsync -aq {self.SRC_PROJECT_PATH}/ {self.PROJECT_PATH}")
        local_project_path = self.SRC_PROJECT_PATH
        os.system(f"rm -r {self.PROJECT_PATH}")
        os.makedirs(self.PROJECT_PATH, exist_ok=True)
        for name in ['Config', 'Saved', 'DerivedDataCache', 'Blank.uproject', 'Intermediate']:
            if os.path.exists(f"{local_project_path}/{name}"):
                os.system(f"rsync -aq {local_project_path}/{name} {self.PROJECT_PATH}/ --exclude 'PipInstall'")
        for name in ['Intermediate/PipInstall', 'ddc']:
            if os.path.exists(f"{local_project_path}/{name}"):
                os.system(f"ln -s {local_project_path}/{name} {self.PROJECT_PATH}/{name}")
            
        if not os.path.exists(f"{self.PROJECT_PATH}/Content"):
            os.makedirs(f"{self.PROJECT_PATH}/Content", exist_ok=True)
        if self.AVATAR_ASSETS_PATH is not None:
            os.system(f"ln -s {self.AVATAR_ASSETS_PATH}/* {self.PROJECT_PATH}/Content")
        with open("/tmp/log", "w") as f:
            subprocess.run(f"rsync -aq \"{self.ASSETS_PATH}/{scene_name}\" {self.PROJECT_PATH}/Content", shell=True, stdout=f, stderr=f)

    def save_ddc(self):
        os.system(f"rsync -aq {self.PROJECT_PATH}/Saved {self.SRC_PROJECT_PATH}/")
        os.system(f"rsync -aq {self.PROJECT_PATH}/Intermediate {self.SRC_PROJECT_PATH}/")
        os.system(f"rsync -aq /data/ddc {self.SRC_PROJECT_PATH}/")
        print("Finished! Saved to ", f"{self.SRC_PROJECT_PATH}/")

    def load_ddc(self):
        pass