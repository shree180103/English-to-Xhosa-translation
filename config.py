from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 10**-5,
        "seq_len": 110,
        "d_model": 512,
        "datasource": 'en_xho',
        "lang_src": "en",
        "lang_tgt": "xh",
        "model_folder": "weights_EN-XH",
        "model_basename": "tmodel_",
        "preload":'latest',
        "tokenizer_file": "tokenizer_{0}.model",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])