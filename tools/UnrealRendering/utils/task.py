import os
import sys
import time
import subprocess


def check_ps(name="UnrealEditor-Cmd"):
    """使用 ps 命令检查 Unreal 进程"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', name],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            pids = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 1:
                    pids.append(int(parts[0]))
            print(pids)
            return True, pids
    except Exception as e:
        print(f"ps 命令出错: {e}")
    return False, []

def run_render_ddc(unreal_engine_path, project_path, logs_path, render_script, scene_name, level_name, scene_path, display=False):
    
    # ========================= !!!!!!!!!!! MUST USE -Cmd ELSE NO COLLISON 
    pod_name = os.environ.get('POD_NAME', 'Pod')
    task_name = str(os.path.basename(render_script.split(' ')[0])).removesuffix(".py")
    log_file_path = os.path.join(logs_path,  f'{task_name}_{scene_name}_{level_name}_{pod_name}.log')
    print(f"Logs to {log_file_path}", flush=True)
    GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    print("Render on GPU:", GPU_ID)
    cmd = [
        f"{unreal_engine_path}",
        f"{os.path.join(project_path, 'Blank.uproject')}",
        scene_path, 
        f"-graphicsadapter={GPU_ID}",
        "-DDC=ProjectLocal",
        "-NoDDCMaintenance",   # 禁用 DDC 维护
        "-NoTextureStreaming", # 直接加载最高精度贴图，避免启动后的后续加载
        "-ShaderCompilerFromDDC",
        "-unattended",
        "-nosplash",
        "-ForceRes",
        "-ResX=1920",
        "-ResY=1080",
        "-Windowed",
        f"-ExecutePythonScript={render_script}",
        f"-UserDir={project_path}", 
        f"-LocalAppConfigDir={os.path.join(project_path,'Epic')}",
        f"-ABSLog={log_file_path}"
    ]
    env={
        "XDG_CONFIG_HOME":"/data/.config",
        "XDG_DATA_HOME":"/data/share",
        "HOME": project_path,
        "UE_LocalDataCachePath": "/data/ddc"
        }
    if not display:
        cmd.append("-RenderOffscreen")
    else:
        env['DISPLAY'] = ":23"
    print(" ".join(cmd))
    
    subprocess.run(cmd, env=env,
                    stdout=subprocess.DEVNULL,  # 屏蔽标准输出
                    stderr=subprocess.DEVNULL, text=True)
    print(f"Render Done. Logs to {log_file_path}", flush=True)


def run_navmesh(unreal_engine_path, project_path, logs_path, render_script, scene_name, level_name, scene_path, display=False):
    
    # ========================= !!!!!!!!!!! MUST USE -Cmd ELSE NO COLLISON 
    pod_name = os.environ.get('POD_NAME', 'Pod')
    log_file_path = os.path.join(logs_path,  f'navmesh_{scene_name}_{level_name}_{pod_name}.log')
    print(f"Logs to {log_file_path}", flush=True)
    GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    print("Render on GPU:", GPU_ID)
    cmd = [
        f"{unreal_engine_path}",
        f"{os.path.join(project_path, 'Blank.uproject')}",
        scene_path, 
        f"-graphicsadapter={GPU_ID}",
        "-DDC=ProjectLocal",
        "-NoDDCMaintenance",   # 禁用 DDC 维护
        "-NoTextureStreaming", # 直接加载最高精度贴图，避免启动后的后续加载
        "-ShaderCompilerFromDDC",
        "-unattended",
        "-nosplash",
        "-ForceRes",
        "-ResX=1920",
        "-ResY=1080",
        "-Windowed",
        f"-ExecutePythonScript={render_script}",
        f"-UserDir={project_path}", 
        f"-LocalAppConfigDir={os.path.join(project_path,'Epic')}",
        f"-ABSLog={log_file_path}"
    ]
    env={
        "XDG_CONFIG_HOME":"/data/.config",
        "XDG_DATA_HOME":"/data/share",
        "HOME": project_path,
        "UE_LocalDataCachePath": "/data/ddc"
        }
    if not display:
        cmd.append("-RenderOffscreen")
    else:
        env['DISPLAY'] = ":23"
    print(" ".join(cmd))
    
    subprocess.run(cmd, env=env,
                    stdout=subprocess.DEVNULL,  # 屏蔽标准输出
                    stderr=subprocess.DEVNULL, text=True)
    print(f"Render Done. Logs to {log_file_path}", flush=True)

def run_render(unreal_engine_path, project_path, logs_path, render_script, scene_name, level_name, scene_path, display=False):
    
    # ========================= !!!!!!!!!!! MUST USE -Cmd ELSE NO COLLISON 
    pod_name = os.environ.get('POD_NAME', 'Pod')
    log_file_path = os.path.join(logs_path, f'render_{scene_name}_{level_name}_{pod_name}.log')
    print(f"Logs to {log_file_path}", flush=True)

    cmd = [
        f"{os.path.abspath(unreal_engine_path)}",
        f"{os.path.join(os.path.abspath(project_path), 'Blank.uproject')}",
        scene_path, 
        f"-graphicsadapter=0",
        "-unattended",
        "-nosplash",
        "-ForceRes",
        "-ResX=1920",
        "-ResY=1080",
        "-Windowed",
        f"-ExecutePythonScript={render_script}",
        f"-UserDir={project_path}", 
        f"-LocalAppConfigDir={project_path}/Epic",
        f"-ABSLog={log_file_path}"
    ]
    if not display:
        cmd.append("-RenderOffscreen")

    print(" ".join(cmd))
    env={
        "XDG_CONFIG_HOME":"/data/.config",
        "XDG_DATA_HOME":"/data/share",
        "HOME": project_path
        }
    subprocess.run(cmd,
                    stdout=subprocess.DEVNULL,  # 屏蔽标准输出
                    stderr=subprocess.DEVNULL, text=True, shell=True)
    print(f"Render Done. Logs to {log_file_path}", flush=True)

def run_prerender_ddc(unreal_engine_path, project_path, logs_path, render_script, scene_name, level_name, scene_path):
    
    # ========================= !!!!!!!!!!! MUST USE -Cmd ELSE NO COLLISON 
    pod_name = os.environ.get('POD_NAME', 'Pod')
    log_file_path = os.path.join(logs_path,  f'prerender_{scene_name}_{level_name}_{pod_name}.log')
    print(f"Logs to {log_file_path}", flush=True)
    GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    print("Render on GPU:", GPU_ID)
    cmd = [
        f"{unreal_engine_path}",
        f"{os.path.join(project_path, 'Blank.uproject')}",
        f"-graphicsadapter={GPU_ID}",
        "-DDC=ProjectLocal",
        "-ShaderCompilerFromDDC",
        "-unattended",
        "-nosplash",
        "-ForceRes",
        "-ResX=1920",
        "-ResY=1080",
        "-Windowed",
        f"-ExecutePythonScript={render_script}",
        f"-UserDir={project_path}", 
        f"-LocalAppConfigDir={os.path.join(project_path,'Epic')}",
        f"-ABSLog={log_file_path}"
    ]
    env={
        "XDG_CONFIG_HOME":"/data/.config",
        "XDG_DATA_HOME":"/data/share",
        "HOME": project_path,
        "UE_LocalDataCachePath": "/data/ddc"
        }
    cmd.append("-RenderOffscreen")
    print(" ".join(cmd))
    
    subprocess.run(cmd, env=env,
                    stdout=subprocess.DEVNULL,  # 屏蔽标准输出
                    stderr=subprocess.DEVNULL, text=True)
    print(f"Prerender Done. Logs to {log_file_path}", flush=True)

def run_cook(unreal_engine_path, project_path, logs_path, scene_name, level_name, scene_path):
    pod_name = os.environ.get('POD_NAME', 'Pod')
    log_file_path = os.path.join(logs_path,  f'cook_{scene_name}_{level_name}_{pod_name}.log')
    print(f"Logs to {log_file_path}", flush=True)

    cmd = [
        f"{unreal_engine_path}",
        f"{os.path.join(project_path, 'Blank.uproject')}",
        # scene_path,
        "-run=DerivedDataCache",
        f"-graphicsadapter=0",
        "-unattended",
        "-nopause",
        "-iterate",
        "-DDC=ProjectLocal",
        "-fill",
        "-Full",
        "-PrecompileShaders",
        "-LogPSO",
        "-TargetPlatform=Linux",
        "-AllowAllShaderFormats",         # 强制包含所有启用的 RHI 格式
        f"-UserDir={project_path}", 
        f"-LocalAppConfigDir={os.path.join(project_path,'Epic')}",
        f"-ABSLog={log_file_path}"
    ]

    
    env={
            "XDG_CONFIG_HOME":"/data/.config",
            "XDG_DATA_HOME":"/data/share",
            "HOME": project_path
        }
    subprocess.run(cmd, env=env,
                    stdout=subprocess.DEVNULL,  # 屏蔽标准输出
                    stderr=subprocess.DEVNULL, text=True)
    print(f"Cook Done. Logs to {log_file_path}", flush=True)

def build_template(project_path):
    os.system(f"sudo mkdir /data")
    os.system(f"sudo chmod -R +777 /data")
    os.makedirs(f"{project_path}/Content")
    if os.path.exists(f"{project_path}"):
        os.system(f"sudo chmod -R +777 {project_path}")
    else:
        os.makedirs(project_path, exist_ok=True)