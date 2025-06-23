import tempfile
import gradio as gr
import torch
import os
import threading
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import queue
import concurrent.futures
import imageio
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
import subprocess

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.




MAX_USERS = 20

BATCH_INTERVAL = 3  # Batch requests every N seconds
MAX_BATCH_SIZE = 16    # set to None for unlimited
# Thread pool for batch processing
BATCH_WORKER_THREADS = 8  # Adjust based on your hardware
batch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_WORKER_THREADS)

DELETE_AFTER_SECONDS = 120  # seconds to delete generated files after creation

# Get available GPU IDs from CUDA_VISIBLE_DEVICES or default to all
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if cuda_visible:
    # Use logical ids for torch
    NUM_VISIBLE = len([x for x in cuda_visible.split(",") if x.strip()])
    GPU_IDS = list(range(NUM_VISIBLE))
    # For nvidia-smi, we still need the physical ids for utilization
    PHYSICAL_IDS = [int(x) for x in cuda_visible.split(",") if x.strip()]
else:
    GPU_IDS = list(range(torch.cuda.device_count()))
    PHYSICAL_IDS = GPU_IDS

# Load a pipeline per visible GPU (logical indices)
pipelines = []
for gpu_id in GPU_IDS:
    pipe = TrellisImageTo3DPipeline.from_pretrained("jetx/trellis-image-large")
    pipe.to(f"cuda:{gpu_id}")
    pipelines.append(pipe)

# Get GPU utilization (use PHYSICAL_IDS for nvidia-smi, but return in logical order)
def get_gpu_utilization():
    try:
        result = subprocess.check_output([
            'nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'
        ])
        util_all = [int(x) for x in result.decode().strip().split('\n')]
        util = [util_all[pid] for pid in PHYSICAL_IDS]
        return util
    except Exception:
        return [0] * len(GPU_IDS)

# Scheduler: pick the GPU with the least utilization (logical index)
def pick_least_busy_gpu():
    util = get_gpu_utilization()
    print(f"Current GPU utilization: {util}")
    return int(min(range(len(GPU_IDS)), key=lambda i: util[i]))

# Load a pipeline from a model folder or a Hugging Face model hub.
torch.set_float32_matmul_precision("medium")
pipeline = TrellisImageTo3DPipeline.from_pretrained("jetx/trellis-image-large")
pipeline.cuda()

# Thread-safe queue for batching
request_queue = queue.Queue()

# NOTE: The generated files (3D asset and videos) will be deleted from the server after 2 minutes. Please download them promptly.
def schedule_delete(filepath, delay=DELETE_AFTER_SECONDS):
    def delete_file():
        time.sleep(delay)
        try:
            os.remove(filepath)
        except Exception:
            pass
    threading.Thread(target=delete_file, daemon=True).start()

def cpu_postprocess(video_gaussian, video_mesh, ply_data, temp_mp4_gaussian, temp_mp4_mesh, temp_ply):
    # Save gaussian video
    imageio.mimsave(temp_mp4_gaussian, video_gaussian, fps=30)
    # Save mesh video
    imageio.mimsave(temp_mp4_mesh, video_mesh, fps=30)
    # Save ply file
    ply_data.save_ply(temp_ply)
    return temp_mp4_gaussian, temp_mp4_mesh, temp_ply


def process_batch(batch, futures):
    try:
        gpu_id = pick_least_busy_gpu()
        pipeline = pipelines[gpu_id]
        with torch.cuda.device(gpu_id):
            outputs_dict = pipeline.run(image=batch, seed=42, num_samples=len(batch), formats=['gaussian'])
            outputs_list = []
            for i in range(len(batch)):
                outputs_list.append({key: [outputs_dict[key][i]] for key in outputs_dict.keys()})
            # Parallelize postprocessing for each batch item using multiprocessing
            for i, outputs in enumerate(outputs_list):
                files_to_download, mp4_files = generate_assets_from_outputs(outputs)
                futures[i].set_result((files_to_download, mp4_files))
    except Exception as e:
        for fut in futures:
            fut.set_exception(e)

def batch_worker():
    while True:
        batch = []
        futures = []
        start_time = time.time()
        try:
            req = request_queue.get(timeout=BATCH_INTERVAL)
            batch.append(req['input'])
            futures.append(req['future'])
        except queue.Empty:
            continue
        while (MAX_BATCH_SIZE is None or len(batch) < MAX_BATCH_SIZE) and (time.time() - start_time < BATCH_INTERVAL):
            try:
                req = request_queue.get_nowait()
                batch.append(req['input'])
                futures.append(req['future'])
            except queue.Empty:
                time.sleep(0.01)
        # Submit the batch to the thread pool for processing
        batch_executor.submit(process_batch, batch, futures)


def generate_assets_from_outputs(outputs, use_process_pool=False, generate_mesh=False):
    mp4_files = []
    files_to_download = []
    video_gaussian = render_utils.render_video(outputs['gaussian'][0])['color']
    ply_data = outputs['gaussian'][0]
    
    if generate_mesh:
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_mp4_gaussian, \
         tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_ply:
        if generate_mesh:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_mp4_mesh:
                if use_process_pool:
                    with ProcessPoolExecutor() as executor:
                        future = executor.submit(cpu_postprocess, video_gaussian, video_mesh, ply_data, temp_mp4_gaussian.name, temp_mp4_mesh.name, temp_ply.name)
                        result = future.result()
                else:
                    imageio.mimsave(temp_mp4_gaussian.name, video_gaussian, fps=30)
                    imageio.mimsave(temp_mp4_mesh.name, video_mesh, fps=30)
                    ply_data.save_ply(temp_ply.name)
                    result = (temp_mp4_gaussian.name, temp_mp4_mesh.name, temp_ply.name)
                mp4_files.extend([result[0], result[1]])
                files_to_download.append(result[2])
                schedule_delete(result[0])
                schedule_delete(result[1])
                schedule_delete(result[2])
        else:
            # Only gaussian video and ply
            if use_process_pool:
                def cpu_postprocess_gaussian(video_gaussian, ply_data, temp_mp4_gaussian, temp_ply):
                    imageio.mimsave(temp_mp4_gaussian, video_gaussian, fps=30)
                    ply_data.save_ply(temp_ply)
                    return temp_mp4_gaussian, temp_ply
                with ProcessPoolExecutor() as executor:
                    future = executor.submit(cpu_postprocess_gaussian, video_gaussian, ply_data, temp_mp4_gaussian.name, temp_ply.name)
                    result = future.result()
            else:
                imageio.mimsave(temp_mp4_gaussian.name, video_gaussian, fps=30)
                ply_data.save_ply(temp_ply.name)
                result = (temp_mp4_gaussian.name, temp_ply.name)
            mp4_files.append(result[0])
            files_to_download.append(result[1])
            schedule_delete(result[0])
            schedule_delete(result[1])

    return files_to_download, mp4_files

# Start the batch worker thread
threading.Thread(target=batch_worker, daemon=True).start()

def gradio_interface(image):
    if image is None:
        return None, []
    fut = concurrent.futures.Future()
    request_queue.put({'input': image, 'future': fut})
    try:
        files_to_download, mp4_files = fut.result()
    except Exception as e:
        return None, [f"Error: {e}"]
    video_gallery = [(path, None) for path in mp4_files]
    if files_to_download:
        return files_to_download[0], video_gallery
    else:
        return None, video_gallery

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--process-pool', action='store_true', help='Use process pool for CPU postprocessing')
    parser.add_argument('--generate-mesh', action='store_true', help='Generate mesh video output (default: only gaussian)')
    parser.add_argument('--port', type=int, default=None, help='Port to run the Gradio server on')
    args = parser.parse_args()

    # Determine port: priority order - CLI arg, env PORT, fallback 23333
    port = args.port or int(os.environ.get('PORT', 0)) or 23333

    iface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.File(label="Download 3D Asset (.ply)", interactive=False),
            gr.Gallery(label="Generated Videos", show_label=True, elem_id="video_gallery", interactive=False, columns=2, show_download_button=True, show_fullscreen_button=True, show_share_button=True)
        ],
        live=True,
        # concurrency_limit = None,
        description="Upload an image to generate a 3D model (.ply) and MP4 files, which you can download or view.\n\nNOTE: The generated files (3D asset and videos) will be deleted from the server within 2 minutes upon creation. Please download them promptly."
    )

    iface.queue(default_concurrency_limit=MAX_USERS)
    
    # Launch Gradio with AzureML-friendly settings
    iface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        inbrowser=False
    )