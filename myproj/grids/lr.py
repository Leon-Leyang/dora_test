from dora import Explorer, Launcher


@Explorer
def explorer(launcher: Launcher):
    launcher.slurm_(
        gpus=2,
        partition="gpu",
        time="720",
        cpus_per_gpu=2,
        setup=['module load miniconda3/23.11.0s', 'conda activate bs'],
    )

    # Example iterating over multiple learning rates and batch sizes
    for lr in [0.001, 0.01, 0.1]:
        sub = launcher.bind([f'optimizer.lr={lr}'])
        sub()
