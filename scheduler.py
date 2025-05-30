import argparse
import itertools
import os
import subprocess
from math import ceil
import torch

def get_available_gpus():
    return list(range(torch.cuda.device_count()))

def chunk_gpus(gpus, num_chunks):
    avg = ceil(len(gpus) / num_chunks)
    return [gpus[i * avg:(i + 1) * avg] for i in range(num_chunks)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated dataset names")
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to store output JSONs")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    models = args.models.split(",")
    datasets = args.datasets.split(",")
    tasks = list(itertools.product(models, datasets))

    available_gpus = get_available_gpus()
    print(f"[Scheduler] Available GPUs: {available_gpus}")

    if len(tasks) > len(available_gpus):
        raise ValueError(f"[Error] Too many tasks ({len(tasks)}) for available GPUs ({len(available_gpus)}).")

    gpu_chunks = chunk_gpus(available_gpus, len(tasks))
    processes = []

    for idx, ((model, dataset), gpu_ids) in enumerate(zip(tasks, gpu_chunks)):
        gpu_ids_str = ",".join(map(str, gpu_ids))

        print(f"[Launch] Task {idx} | Model: {model} | Dataset: {dataset} | GPUs: {gpu_ids_str}")

        cmd = [
            "python", "multi_gpu_runner.py",
            "--model_name", model,
            "--dataset_name", dataset,
            "--template_path", args.template_path,
            "--output_path", args.output_folder,
            "--gpu_ids", gpu_ids_str
        ]
        p = subprocess.Popen(cmd)
        processes.append(p)

    for p in processes:
        p.wait()

    print("[Scheduler] All subtasks finished.")

if __name__ == "__main__":
    main()
