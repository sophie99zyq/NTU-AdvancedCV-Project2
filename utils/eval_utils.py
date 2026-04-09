import json
import os


def save_results(results_dict, dataset_name, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{dataset_name.replace('→', '_to_')}.json")
    with open(path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {path}")


def load_results(dataset_name, save_dir='./results'):
    path = os.path.join(save_dir, f"{dataset_name.replace('→', '_to_')}.json")
    with open(path, 'r') as f:
        return json.load(f)
