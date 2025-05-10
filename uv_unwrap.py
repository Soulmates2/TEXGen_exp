import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj, IO
from pytorch3d.renderer import (
	TexturesUV,
)

# Load obj mesh, rescale the mesh to fit into the bounding box
def load_obj_mesh(mesh_path, scale_factor=1.0, auto_center=True, autouv=False, device="cpu"):
    mesh = load_objs_as_meshes([mesh_path], device=device)
    if auto_center:
        verts = mesh.verts_packed()
        max_bb = (verts - 0).max(0)[0]
        min_bb = (verts - 0).min(0)[0]
        scale = (max_bb - min_bb).max()/2
        center = (max_bb+min_bb) /2
        mesh.offset_verts_(-center)
        mesh.scale_verts_((scale_factor / float(scale)))
    else:
        mesh.scale_verts_((scale_factor))

    if autouv or (mesh.textures is None):
        mesh = uv_unwrap(mesh)
    return mesh


def load_glb_mesh(mesh_path, scale_factor=1.0, auto_center=True, autouv=False, device="cpu"):
    from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    with open(mesh_path, "rb") as f:
        mesh = io.load_mesh(f, include_textures=True, device=device)
    if auto_center:
        verts = mesh.verts_packed()
        max_bb = (verts - 0).max(0)[0]
        min_bb = (verts - 0).min(0)[0]
        scale = (max_bb - min_bb).max()/2 
        center = (max_bb+min_bb) /2
        mesh.offset_verts_(-center)
        mesh.scale_verts_((scale_factor / float(scale)))
    else:
        mesh.scale_verts_((scale_factor))

    if autouv or (mesh.textures is None):
        mesh = uv_unwrap(mesh)
    return mesh


def uv_unwrap(mesh, sampling_mode="bilinear"):
    verts_list = mesh.verts_list()[0]
    faces_list = mesh.faces_list()[0]

    import xatlas
    v_np = verts_list.cpu().numpy()
    f_np = faces_list.int().cpu().numpy()
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 4
    atlas.generate(chart_options=chart_options)
    vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

    vt = torch.from_numpy(vt_np.astype(np.float32)).type(verts_list.dtype).to(mesh.device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).type(faces_list.dtype).to(mesh.device)

    new_map = torch.zeros((1024, 1024, 3), device=mesh.device)
    new_tex = TexturesUV(
        [new_map], 
        [ft], 
        [vt], 
        sampling_mode=sampling_mode
        )

    mesh.textures = new_tex
    return mesh


def save_mesh(save_path, mesh):
    save_obj(save_path, 
            mesh.verts_list()[0],
            mesh.faces_list()[0],
            verts_uvs=mesh.textures.verts_uvs_list()[0],
            faces_uvs=mesh.textures.faces_uvs_list()[0],
            texture_map=mesh.textures.maps_list()[0])

if __name__ == "__main__":
    import os
    from glob import glob

    # obj_path_list = sorted(glob("../dataset/data_objaverse/*/model.obj")) + sorted(glob("../dataset/data_objaverse_xl/*/model.obj"))
    # obj_path_list = sorted(glob("../dataset/teaser/*/model.obj"))
    # obj_path_list = sorted(glob("../../dataset/dataset_final_0404/*/model.obj"))
    obj_path_list = sorted(glob("../../dataset/qualitative/*/model.obj"))
    print(f"the number of obj files: {len(obj_path_list)}")

    texgen_config_file = open("assets/input_list/test_input_qualitative.jsonl", "w")

    for i, obj_path in enumerate(obj_path_list):
        print(f"================ {i}/{len(obj_path_list)} ================")
        model_id = obj_path.split('/')[-2]
        mesh = load_obj_mesh(obj_path, auto_center=False, autouv=True)
        save_dir = f"assets/models_qualitative/{model_id[:2]}/{model_id}"
        # os.makedirs(save_dir, exist_ok=True)
        save_path = f"assets/models_qualitative/{model_id[:2]}/{model_id}/model.obj"
        # save_mesh(save_path, mesh)
        print(f"Save at {save_path}")
        with open(obj_path.replace("model.obj", "config.yaml"), "r") as f:
            content = f.read()
            prompt = content.split("prompt: ")[1].split("\n")[0]
            if prompt[0] == "'":
                prompt = '"' + prompt[1:]
                print(prompt)
            if prompt[-1] == "'":
                prompt = prompt[:-1] + '"'
                print(prompt)
        texgen_config_file.write("{" + f"\"id\": \"{model_id}\", \"result\": {prompt}, \"root_dir\": \"assets/models_qualitative\"" + "}\n")
    texgen_config_file.close()

    # cmd = 'python launch.py --config configs/texgen_test.yaml --test --gpu 0 data.eval_scene_list="assets/input_list/test_input.jsonl" exp_root_dir=outputs_test name=teaser tag=test system.weights="assets/checkpoints/texgen_v1.ckpt"'
    # print(cmd)
    # os.system(cmd)