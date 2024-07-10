import hydra
from omegaconf import DictConfig
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import get_model
from dora.distrib import init


def train(cfg):
    # Initialize distributed environment
    distrib = init()

    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.CIFAR10(root=cfg.dataset, train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset) if cfg.trainer.distributed else None
    dataloader = DataLoader(dataset, batch_size=cfg.trainer.batch_size, sampler=sampler)

    # Model setup
    model = get_model(cfg).to(distrib.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum)

    # Training loop
    model.train()
    for epoch in range(cfg.trainer.epochs):
        if cfg.trainer.distributed:
            sampler.set_epoch(epoch)
        for inputs, labels in dataloader:
            inputs = inputs.to(distrib.device)
            labels = labels.to(distrib.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if distrib.is_master:
            print(f'Epoch {epoch+1}/{cfg.trainer.epochs} - Loss: {loss.item()}')

    # Ensure synchronization of all processes
    distrib.barrier()


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
