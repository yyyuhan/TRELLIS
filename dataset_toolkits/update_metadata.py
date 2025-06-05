# Update metadata for each stage of data rendering

import os
import sys
import argparse
import pandas as pd

def register_rendered_data(output_dir):
    """
    Register the rendered data in the metadata file.
    """
    metadata_file = os.path.join(output_dir, 'metadata.csv')
    rendered_file = os.path.join(output_dir, 'rendered_0.csv')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    # Register the rendered data under renders/sha256 to the rendered file
    if not os.path.exists(rendered_file):
        raise FileNotFoundError(f"Rendered file {rendered_file} does not exist.")
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


def register_voxelized_data(output_dir):
    """
    Register the voxelized data in the metadata file.
    """
    metadata_file = os.path.join(output_dir, 'metadata.csv')
    voxelized_file = os.path.join(output_dir, 'voxelized_0.csv')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    # Register the voxelized data under voxels/sha256 to the voxelized file
    if not os.path.exists(voxelized_file):
        raise FileNotFoundError(f"Voxelized file {voxelized_file} does not exist.")
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

def register_dinov2_feature_data(output_dir):
    """
    Register the DINOv2 feature data in the metadata file.
    """
    metadata_file = os.path.join(output_dir, 'metadata.csv')
    dinov2_file = os.path.join(output_dir, 'feature_dinov2_vitl14_reg_0.csv')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    # Register the DINOv2 feature data under dinov2_features/sha256 to the dinov2 file
    if not os.path.exists(dinov2_file):
        raise FileNotFoundError(f"DINOv2 feature file {dinov2_file} does not exist.")
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

def register_stage2_generation_latent_data(output_dir):
    """
    Register the stage 2 generation latent data in the metadata file.
    """
    key = 'latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16'
    metadata_file = os.path.join(output_dir, 'metadata.csv')
    stage2_file = os.path.join(output_dir, 'latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16_0.csv')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    # Register the stage 2 generation latent data under ss_latents/sha256 to the stage2 file
    if not os.path.exists(stage2_file):
        raise FileNotFoundError(f"Stage 2 generation latent file {stage2_file} does not exist.")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update metadata for rendered data.")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing the output data.')
    args = parser.parse_args()

    output_dir = args.output_dir

    # Register rendered data
    # register_rendered_data(output_dir)
    # Register voxelized data
    # register_voxelized_data(output_dir)
    # Register DINOv2 feature data
    # register_dinov2_feature_data(output_dir)
    # Register stage 2 generation data
    register_stage2_generation_latent_data(output_dir)