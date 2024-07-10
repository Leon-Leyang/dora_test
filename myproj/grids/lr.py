from dora import Explorer, Launcher


@Explorer
def explorer(launcher: Launcher):
    launcher.slurm_(
        gpus=2,
        mem_per_gpu=32000,
        partition="gpu",
        time="12:00:00",
        job_name="lr-grid",
        mail_type='ALL',
        mail_user='leyang_hu@brown.edu'
    )

    # Example iterating over multiple learning rates and batch sizes
    for lr in [0.001, 0.01, 0.1]:
        sub = launcher.bind([f'optimizer.lr={lr}'])
        sub.slurm_(
            output=f"logs/lr-{lr}.out",
            error=f"logs/lr-{lr}.out"
        )
        sub()
