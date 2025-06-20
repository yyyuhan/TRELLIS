import argparse
import copy
from functools import partial
import importlib
import json
import os
import sys
import pandas as pd
import utils3d
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import open3d as o3d
from subprocess import DEVNULL, call

from utils import sphere_hammersley_sequence
import trellis.models as models

METADATA_FILENAME = 'metadata.csv'

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

torch.set_grad_enabled(False)

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    # Multiview rendering parameters
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    # Image embedding
    parser.add_argument('--feat_model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    # SS parameters
    parser.add_argument('--ss_enc_pretrained', type=str, default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
                        help='Pretrained sparse structure encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of in-house models')
    parser.add_argument('--ss_enc_self', type=str, default=None,
                        help='Self-defined sparse structure encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ss_enc_self_ckpt', type=str, default=None,
                        help='Self-defined sparse sturcture checkpoint to load')
    parser.add_argument('--ss_resolution', type=int, default=64,
                        help='Resolution of voxel grid to encode')
    # SLAT parameters
    parser.add_argument('--slat_enc_pretrained', type=str, default='microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--slat_enc_self', type=str, default=None,
                        help='Slat encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--slat_enc_self_ckpt', type=str, default=None,
                        help='SLAT encoding checkpoint to load')
    # Data split
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of the process in distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of processes in distributed training')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Maximum number of worker threads for rendering')
    return parser.parse_args(args)

def check_common_inputs(output_dir, output_filename, metadata_filename=METADATA_FILENAME):
    """
    Check if the output directory and metadata file exist.
    """
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist.")
    metadata_file = os.path.join(output_dir, metadata_filename)
    if not os.path.exists(metadata_file):
        raise ValueError(f"Metadata file {metadata_file} does not exist.")
    output_file = os.path.join(output_dir, output_filename)
    if not os.path.exists(output_file):
        raise ValueError(f"Output file {output_file} does not exist.")
    return metadata_file, output_file

def _render_multiview(file_path, sha256, output_dir, num_views):
    output_folder = os.path.join(output_dir, 'renders', sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    call(args, stdout=DEVNULL, stderr=DEVNULL)
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'rendered': True}

def _voxelize(file, sha256, output_dir):
    mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'renders', sha256, 'mesh.ply'))
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    utils3d.io.write_ply(os.path.join(output_dir, 'voxels', f'{sha256}.ply'), vertices)
    return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}

def get_voxels(instance, opt):
    position = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{instance}.ply'))[0]
    coords = ((torch.tensor(position) + 0.5) * opt.ss_resolution).int().contiguous()
    ss = torch.zeros(1, opt.ss_resolution, opt.ss_resolution, opt.ss_resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss

def get_image_data(frames, sha256):
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(view):
            image_path = os.path.join(opt.output_dir, 'renders', sha256, view['file_path'])
            try:
                image = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            c2w = torch.tensor(view['transform_matrix'])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = view['camera_angle_x']
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics
            }
        
        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data

def register_rendered_data(output_dir, output_filename, metadata_filename=METADATA_FILENAME):
    """
    Register the rendered data in the metadata file.
    """
    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)
    metadata_file, rendered_file = check_common_inputs(output_dir, output_filename, metadata_filename)
    # metadata_file = os.path.join(output_dir, metadata_filename)
    # rendered_file = os.path.join(output_dir, output_filename)
    # if not os.path.exists(metadata_file):
    #     raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    # if not os.path.exists(rendered_file):
    #     raise FileNotFoundError(f"Rendered file {rendered_file} does not exist.")
    # Register the rendered data under renders/sha256 to the rendered file
    rendered_df = pd.read_csv(rendered_file)
    new_rendered_rows = []
    for sha256 in os.listdir(os.path.join(output_dir, 'renders')):
        if sha256 not in rendered_df['sha256'].values:
            new_row = {'sha256': sha256, 'rendered': True}
            new_rendered_rows.append(new_row)
    if new_rendered_rows:
        new_rendered_df = pd.DataFrame(new_rendered_rows)
        rendered_df = pd.concat([rendered_df, new_rendered_df], ignore_index=True)
    rendered_df.to_csv(rendered_file, index=False)
    print(f"Rendered data registered in {rendered_file}.")

    # Update the metadata file with rendered data in place by turning the rendered column to True
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    if not os.path.exists(os.path.join(output_dir, 'renders')):
        print(f"Warning: Renders directory {os.path.join(output_dir, 'renders')} does not exist.")
        return
    # Read the metadata file
    metadata_df = pd.read_csv(metadata_file)
    # Update the rendered column to True for the sha256s in the renders directory
    for sha256 in os.listdir(os.path.join(output_dir, 'renders')):
        if sha256 in metadata_df['sha256'].values:
            metadata_df.loc[metadata_df['sha256'] == sha256, 'rendered'] = True
            print(f"Updated metadata for {sha256} to rendered=True.")
        else:
            print(f"Warning: {sha256} not found in metadata file.")
    metadata_df.to_csv(metadata_file, index=False)
    print("Metadata updated with rendered data.")

def register_voxelized_data(output_dir, output_filename, metadata_filename=METADATA_FILENAME):
    """
    Register the voxelized data in the metadata file.
    """
    metadata_file, voxelized_file = check_common_inputs(output_dir, output_filename, metadata_filename)
    # metadata_file = os.path.join(output_dir, metadata_filename)
    # voxelized_file = os.path.join(output_dir, output_filename)
    # if not os.path.exists(metadata_file):
    #     raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    # # Register the voxelized data under voxels/sha256 to the voxelized file
    # if not os.path.exists(voxelized_file):
    #     raise FileNotFoundError(f"Voxelized file {voxelized_file} does not exist.")
    voxelized_df = pd.read_csv(voxelized_file)
    voxelized_sha256s = voxelized_df[voxelized_df['voxelized'] == True]['sha256'].values
    metadata_df = pd.read_csv(metadata_file)
    # Update the voxelized column to True for the sha256s in the voxels directory
    for sha256 in voxelized_sha256s:
        if sha256 in metadata_df['sha256'].values:
            metadata_df.loc[metadata_df['sha256'] == sha256, 'voxelized'] = True
            print(f"Updated metadata for {sha256} to voxelized=True.")
        else:
            print(f"Warning: {sha256} not found in metadata file.")
    metadata_df.to_csv(metadata_file, index=False)

def register_dinov2_feature_data(output_dir, output_filename, metadata_filename=METADATA_FILENAME):
    """
    Register the DINOv2 feature data in the metadata file.
    """
    metadata_file, dinov2_file = check_common_inputs(output_dir, output_filename, metadata_filename)
    # metadata_file = os.path.join(output_dir, 'metadata.csv')
    # dinov2_file = os.path.join(output_dir, 'feature_dinov2_vitl14_reg_0.csv')
    # if not os.path.exists(metadata_file):
    #     raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    # if not os.path.exists(dinov2_file):
    #     raise FileNotFoundError(f"DINOv2 feature file {dinov2_file} does not exist.")
    # Register the DINOv2 feature data under dinov2_features/sha256 to the dinov2 file
    dinov2_df = pd.read_csv(dinov2_file)
    dinov2_sha256s = dinov2_df[dinov2_df['feature_dinov2_vitl14_reg'] == True]['sha256'].values
    metadata_df = pd.read_csv(metadata_file)
    # Update the dinov2_feature column to True for the sha256s in the dinov2_features directory
    for sha256 in dinov2_sha256s:
        if sha256 in metadata_df['sha256'].values:
            # Suppose column feature_dinov2_vitl14_reg does not exist
            if 'feature_dinov2_vitl14_reg' not in metadata_df.columns:
                metadata_df['feature_dinov2_vitl14_reg'] = False
            metadata_df.loc[metadata_df['sha256'] == sha256, 'feature_dinov2_vitl14_reg'] = True
            print(f"Updated metadata for {sha256} to feature_dinov2_vitl14_reg=True.")
        else:
            print(f"Warning: {sha256} not found in metadata file.")
    metadata_df.to_csv(metadata_file, index=False)

def register_slat_data(output_dir, column_key, output_filename, metadata_filename=METADATA_FILENAME):
    """
    Register the stage 2 generation latent data in the metadata file.
    """
    key = column_key
    metadata_file, stage2_file = check_common_inputs(output_dir, output_filename, metadata_filename=METADATA_FILENAME)
    # key = 'latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16'
    # metadata_file = os.path.join(output_dir, 'metadata.csv')
    # stage2_file = os.path.join(output_dir, 'latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16_0.csv')
    # if not os.path.exists(metadata_file):
    #     raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    # # Register the stage 2 generation latent data under ss_latents/sha256 to the stage2 file
    # if not os.path.exists(stage2_file):
    #     raise FileNotFoundError(f"Stage 2 generation latent file {stage2_file} does not exist.")
    stage2_df = pd.read_csv(stage2_file)
    stage2_sha256s = stage2_df[stage2_df[key] == True]['sha256'].values
    metadata_df = pd.read_csv(metadata_file)
    # Update the ss_latent column to True for the sha256s in the ss_latents directory
    for sha256 in stage2_sha256s:
        if sha256 in metadata_df['sha256'].values:
            # Suppose column does not exist
            if key not in metadata_df.columns:
                metadata_df[key] = False
            metadata_df.loc[metadata_df['sha256'] == sha256, key] = True
            print(f"Updated metadata for {sha256} to ss_latent_dinov2_vitl14_reg=True.")
        else:
            print(f"Warning: {sha256} not found in metadata file.")
    metadata_df.to_csv(metadata_file, index=False)

def render_multiview_images(opt):
    """ Perform rendering multiview images following step 4 of the dataset generation pipeline. """
        # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        metadata = metadata[metadata['file_identifier'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' in metadata.columns:
            metadata = metadata[metadata['rendered'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]

    print(f'Rnder Multiview Images: Processing {len(metadata)} objects...')

    # process objects
    func = partial(_render_multiview, output_dir=opt.output_dir, num_views=opt.num_views)
    rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    output_filename = f'rendered_{opt.rank}.csv'
    rendered.to_csv(os.path.join(opt.output_dir, output_filename), index=False)
    register_rendered_data(opt.output_dir, output_filename)


def voxelize_3d_models(opt):
    """ Step 5: Voxelize 3D Models """
    os.makedirs(os.path.join(opt.output_dir, 'voxels'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' not in metadata.columns:
            raise ValueError('metadata.csv does not have "rendered" column, please run "build_metadata.py" first')
        metadata = metadata[metadata['rendered'] == True]
        if 'voxelized' in metadata.columns:
            metadata = metadata[metadata['voxelized'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')):
            pts = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
            records.append({'sha256': sha256, 'voxelized': True, 'num_voxels': len(pts)})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Voxelize 3D Models: Processing {len(metadata)} objects...')

    # process objects
    func = partial(_voxelize, output_dir=opt.output_dir)
    voxelized = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Voxelizing')
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    output_filename = f'voxelized_{opt.rank}.csv'
    voxelized.to_csv(os.path.join(opt.output_dir, output_filename), index=False)
    register_voxelized_data(opt.output_dir, output_filename)


def extract_dino_features(opt):
    """ Step 6: Extract DINO Features """
    feature_name = opt.feat_model
    os.makedirs(os.path.join(opt.output_dir, 'features', feature_name), exist_ok=True)

    # load model
    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if f'feature_{feature_name}' in metadata.columns:
            metadata = metadata[metadata[f'feature_{feature_name}'] == False]
        metadata = metadata[metadata['voxelized'] == True]
        metadata = metadata[metadata['rendered'] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, 'features', feature_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'feature_{feature_name}' : True})
            sha256s.remove(sha256)

    # extract features
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor, \
            ThreadPoolExecutor(max_workers=8) as saver_executor:
            def loader(sha256):
                try:
                    with open(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json'), 'r') as f:
                        metadata = json.load(f)
                    frames = metadata['frames']
                    data = []
                    for datum in get_image_data(frames, sha256):
                        datum['image'] = transform(datum['image'])
                        data.append(datum)
                    positions = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
                    load_queue.put((sha256, data, positions))
                except Exception as e:
                    print(f"Error loading data for {sha256}: {e}")

            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack, patchtokens, uv):
                pack['patchtokens'] = F.grid_sample(
                    patchtokens,
                    uv.unsqueeze(1),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).permute(0, 2, 1).cpu().numpy()
                pack['patchtokens'] = np.mean(pack['patchtokens'], axis=0).astype(np.float16)
                save_path = os.path.join(opt.output_dir, 'features', feature_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'feature_{feature_name}' : True})
                
            for _ in tqdm(range(len(sha256s)), desc="Extracting features"):
                sha256, data, positions = load_queue.get()
                positions = torch.from_numpy(positions).float().cuda()
                indices = ((positions + 0.5) * 64).long()
                assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
                n_views = len(data)
                N = positions.shape[0]
                pack = {
                    'indices': indices.cpu().numpy().astype(np.uint8),
                }
                patchtokens_lst = []
                uv_lst = []
                for i in range(0, n_views, opt.batch_size):
                    batch_data = data[i:i+opt.batch_size]
                    bs = len(batch_data)
                    batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
                    batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
                    batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
                    features = dinov2_model(batch_images, is_training=True)
                    uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
                    patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                    patchtokens_lst.append(patchtokens)
                    uv_lst.append(uv)
                patchtokens = torch.cat(patchtokens_lst, dim=0)
                uv = torch.cat(uv_lst, dim=0)

                # save features
                saver_executor.submit(saver, sha256, pack, patchtokens, uv)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    records = pd.DataFrame.from_records(records)
    output_filename = f'feature_{feature_name}_{opt.rank}.csv'
    records.to_csv(os.path.join(opt.output_dir, output_filename), index=False)
    register_dinov2_feature_data(opt.output_dir, output_filename)

def encode_ss_latents(opt):
    """ Step 7: Encode Sparse Structure Latents """
    if opt.ss_enc_self is None:
        latent_name = f'{opt.ss_enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.ss_enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.ss_enc_self}_{opt.ss_enc_self_ckpt}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.ss_enc_self, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.ss_enc_self, 'ckpts', f'encoder_{opt.ss_enc_self_ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    
    os.makedirs(os.path.join(opt.output_dir, 'ss_latents', latent_name), exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['voxelized'] == True]
        if f'ss_latent_{latent_name}' in metadata.columns:
            metadata = metadata[metadata[f'ss_latent_{latent_name}'] == False]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    
    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, 'ss_latents', latent_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})
            sha256s.remove(sha256)

    # encode latents
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=32) as saver_executor:
            def loader(sha256):
                try:
                    ss = get_voxels(sha256)[None].float()
                    load_queue.put((sha256, ss))
                except Exception as e:
                    print(f"Error loading features for {sha256}: {e}")
            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack):
                save_path = os.path.join(opt.output_dir, 'ss_latents', latent_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})
                
            for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
                sha256, ss = load_queue.get()
                ss = ss.cuda().float()
                latent = encoder(ss, sample_posterior=False)
                assert torch.isfinite(latent).all(), "Non-finite latent"
                pack = {
                    'mean': latent[0].cpu().numpy(),
                }
                saver_executor.submit(saver, sha256, pack)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'ss_latent_{latent_name}_{opt.rank}.csv'), index=False)


def encode_slat(opt):
    if opt.slat_enc_self is None:
        latent_name = f'{opt.feat_model}_{opt.slat_enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.slat_enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.feat_model}_{opt.slat_enc_self}_{opt.slat_enc_self_ckpt}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.slat_enc_self, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.slat_enc_self, 'ckpts', f'encoder_{opt.slat_enc_self_ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    
    os.makedirs(os.path.join(opt.output_dir, 'latents', latent_name), exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            sha256s = [line.strip() for line in f]
        metadata = metadata[metadata['sha256'].isin(sha256s)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata[f'feature_{opt.feat_model}'] == True]
        if f'latent_{latent_name}' in metadata.columns:
            metadata = metadata[metadata[f'latent_{latent_name}'] == False]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    
    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, 'latents', latent_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'latent_{latent_name}': True})
            sha256s.remove(sha256)

    # encode latents
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=32) as saver_executor:
            def loader(sha256):
                try:
                    feats = np.load(os.path.join(opt.output_dir, 'features', opt.feat_model, f'{sha256}.npz'))
                    load_queue.put((sha256, feats))
                except Exception as e:
                    print(f"Error loading features for {sha256}: {e}")
            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack):
                save_path = os.path.join(opt.output_dir, 'latents', latent_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'latent_{latent_name}': True})
                
            for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
                sha256, feats = load_queue.get()
                feats = sp.SparseTensor(
                    feats = torch.from_numpy(feats['patchtokens']).float(),
                    coords = torch.cat([
                        torch.zeros(feats['patchtokens'].shape[0], 1).int(),
                        torch.from_numpy(feats['indices']).int(),
                    ], dim=1),
                ).cuda()
                latent = encoder(feats, sample_posterior=False)
                assert torch.isfinite(latent.feats).all(), "Non-finite latent"
                pack = {
                    'feats': latent.feats.cpu().numpy().astype(np.float32),
                    'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
                }
                saver_executor.submit(saver, sha256, pack)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    records = pd.DataFrame.from_records(records)
    output_filename = f'latent_{latent_name}_{opt.rank}.csv'
    records.to_csv(os.path.join(opt.output_dir, output_filename), index=False)

    register_slat_data(opt.output_dir, f'latent_{latent_name}', output_filename)

if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')
    opt = parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # install blender
    print('Checking blender...', flush=True)
    _install_blender()
    
    try:
        render_multiview_images(opt)
        voxelize_3d_models(opt)
        extract_dino_features(opt)
        encode_ss_latents(opt)
        encode_slat(opt)
    except Exception as e:
        print(f"Error during rendering: {e}", flush=True)
        sys.exit(1)
    