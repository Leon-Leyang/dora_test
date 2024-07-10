import dora.distrib
from dora import hydra_main, get_xp


@hydra_main(
    config_path="./conf",  # path where the config is stored, relative to the parent of `mycode`.
    config_name="config"  # a file `config.yaml` should exist there.
)
def main(cfg):
    dora.distrib.init()

    xp = get_xp()
    xp.cfg  # parsed configuration
    xp.sig  # signature for the current run
    # Hydra run folder will automatically be set to xp.folder!

    xp.link  # link object, can send back metrics to Dora
    # If you load a previous checkpoint, you should always make sure
    # That the Dora Link is consistent with what is in the checkpoint with
    # history = checkpoint['history']
    # xp.link.update_history(history)

    for t in range(10):
        xp.link.push_metrics({"loss": 1/(t + 1)})
