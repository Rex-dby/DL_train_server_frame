import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import deepspeed
import json
from tqdm.auto import tqdm

def args_parser():
    parser = argparse.ArgumentParser(description='Image classification training with DeepSpeed')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset with class folders')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args()

def get_dataloaders(data_dir, image_size, batch_size, local_rank):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Distributed sampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    return dataloader, len(dataset.classes)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)

def train():
    args = args_parser()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')

    dataloader, num_classes = get_dataloaders(args.dataset_path, args.image_size, args.batch_size, local_rank)

    model = Classifier(num_classes).to(local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    ds_config = json.load(open(args.deepspeed_config))

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config,
        optimizer=optimizer
    )

    for epoch in range(args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        if local_rank == 0:
            pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")

        model_engine.train()
        total_loss = 0
        for images, labels in dataloader:
            images = images.to(local_rank)
            labels = labels.to(local_rank)

            outputs = model_engine(images)
            loss = loss_fn(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            if local_rank == 0:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        if local_rank == 0:
            ckpt_dir = os.path.join(args.output_dir, f'epoch_{epoch}')
            os.makedirs(ckpt_dir, exist_ok=True)
            model_engine.save_checkpoint(ckpt_dir)

        torch.distributed.barrier()

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    train()
