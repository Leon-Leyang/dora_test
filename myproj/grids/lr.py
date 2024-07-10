from dora import Explorer, Launcher


@Explorer
def experiment(launcher: Launcher):
    # Example iterating over multiple learning rates and batch sizes
    for lr in [0.001, 0.01, 0.1]:
        launcher(f'optimizer.lr={lr}')
