import os
import sys
SyncMVD_path = "/home/dld/texture/SyncMVD"
sys.path.append(SyncMVD_path)
from glob import glob
import subprocess
import shutil

selected_dict = {
    "0b83ad557c8a47df81d2845dc11d6439": "A post-apocalyptic rusted bus interior with chairs bolted to the floor.",
    "0f1e894aa6264e0aa3b22c1efb82703a": "A luxurious desert tent made of flowing cream-colored fabric, with a brass oil lamp in the center and two low sofas with cushions.",
    "1d3ad2a23c444d5abfabf1a48eeb8c84": "A dwarven forge beer mug carved from volcanic stone with a charred wooden handle, and a glowing red rune pattern swirling inside.",
    "5fd2c9bf3e834e839b26649db7afe4c1": "A clean beige cowboy hat with a bold autograph printed on the inner crown in black ink.",
    "97a2952223444f92aa98e60280b6d637": "A lacquered jade chest with golden hinges and a silk interior embroidered with dragon motifs in gold thread.",
    "1407f16a313a4e7db0cb35831773f510": "A mystic's tent with deep purple silk drapes embroidered in constellations, and a black-lacquered desk topped with glowing runes.",
    "9176a3219f494f779ca8ea08407c4c64": "A cable car adorned with ornate brass filigree patterns across its panels and rich mahogany wood textures polished to a deep sheen.",
    "84670ba2c265404a9220eb35f092e2d1": "A dark-stained wooden forest cabin with hand-carved bunk beds under a dim lantern glow inside.",
    "1762220997d74772ba84ac29d61435d9": "A moss-covered circular stone well with a steep roof, nestled in a quiet forest clearing.",
    "a4f07e22f8364317b3c83291395687d6": "A post-apocalyptic concrete bunker texture with cracks and moss.",
    "bcd03f021ca1465c974862ec401c324a": "A bright red vintage car with weathered tan leather seats and a dashboard featuring brass-accented dials.",
    "bde9edccdc2c477d8d670fc47282249e": "A weathered wooden house with rusted nails, cracked siding."
}

if __name__ == "__main__":
    # obj_path_list = sorted(glob("../../dataset/data_objaverse/*/model.obj")) + sorted(glob("../../dataset/data_objaverse_xl/*/model.obj")) \
    #                 sorted(glob("../RayTexGen_exp2/meshes/new"))
    # print(f"the number of obj files: {len(obj_path_list)}")
    
    
    # projection command
    dataset_name = "dataset_final_0404"
    # dataset_name = "qualitative"
    # config_path_list = sorted(glob(f"../../dataset/{dataset_name}/*/config.yaml"))
    uuid_list = list(selected_dict.keys())
    prompt_list = list(selected_dict.values())
    config_path_list = [f"../../dataset/{dataset_name}/{uuid}/config.yaml" for uuid in uuid_list]
    print(f"the number of config files: {len(config_path_list)}")

    cmd_list = []
    model_id_list = []
    for config_path in config_path_list:
        model_id = config_path.split("/")[-2]
        model_id_list.append(model_id)

        os.makedirs(f"{SyncMVD_path}/output", exist_ok=True)

        cmd = f"python {SyncMVD_path}/run_experiment.py"
        cmd += f" --config {config_path}"
        # cmd += f" --output {SyncMVD_path}/output"
        cmd += f" --output {SyncMVD_path}/output_selected"
        cmd += f" --prompt \"{selected_dict[model_id]}\""
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
        os.makedirs(f"/home/dld/texture/TEXGen/assets/models_selected/{model_id[:2]}/{model_id}", exist_ok=True)
        shutil.copy(f"/home/dld/texture/TEXGen/assets/models_final/{model_id[:2]}/{model_id}/model.obj", f"/home/dld/texture/TEXGen/assets/models_selected/{model_id[:2]}/{model_id}/model.obj")
        shutil.copy(f"/home/dld/texture/TEXGen/assets/models_final/{model_id[:2]}/{model_id}/model.mtl", f"/home/dld/texture/TEXGen/assets/models_selected/{model_id[:2]}/{model_id}/model.mtl")
        shutil.copy(f"{SyncMVD_path}/output_selected/{model_id}/results/textured.png", \
                    f"/home/dld/texture/TEXGen/assets/models_selected/{model_id[:2]}/{model_id}/model.png")
        