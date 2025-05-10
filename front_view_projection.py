import os
import sys
SVP_path = "SingleViewProjection"
sys.path.append(SVP_path)
from glob import glob
import subprocess
import shutil


if __name__ == "__main__":
    # obj_path_list = sorted(glob("../../dataset/data_objaverse/*/model.obj")) + sorted(glob("../../dataset/data_objaverse_xl/*/model.obj")) \
    #                 sorted(glob("../RayTexGen_exp2/meshes/new"))
    # print(f"the number of obj files: {len(obj_path_list)}")
    
    
    # projection command
    dataset_name = "dataset_final_0404"
    # dataset_name = "qualitative"
    config_path_list = sorted(glob(f"../../dataset/{dataset_name}/*/config.yaml"))
    print(f"the number of config files: {len(config_path_list)}")

    cmd_list = []
    model_id_list = []
    for config_path in config_path_list:
        model_id = config_path.split("/")[-2]
        model_id_list.append(model_id)

        os.makedirs(f"{SVP_path}/output", exist_ok=True)

        cmd = f"python {SVP_path}/run_experiment.py"
        cmd += f" --config {config_path}"
        cmd += f" --output {SVP_path}/output"
        cmd += f" --prefix {model_id}"
        cmd += f" --timeformat \'\'"
        cmd_list.append(cmd)
    
    # run and copy the result uv map
    for i, cmd in enumerate(cmd_list):
        print(f"========================= {i+1}/{len(cmd_list)} =========================")
        print(cmd)
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"fail with return code: {result.returncode}. cmd: {cmd}")
            continue
        model_id = model_id_list[i]
        shutil.copy(f"{SVP_path}/output/{model_id}/results/textured.png", \
                    f"/home/dld/texture/TEXGen/assets/models_qualitative/{model_id[:2]}/{model_id}/model.png")
        