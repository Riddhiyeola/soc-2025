import comet_ml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from omniglot_dataset import OmniglotDataset
from model import MetaLearner

num_classes = 5
num_shots = 5
num_queries = 5
num_epochs = 10
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

train_dataset = OmniglotDataset(root_dir="path/to/omniglot/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=num_classes, shuffle=True)

meta_learner = MetaLearner(num_classes, num_shots, num_queries)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(meta_learner.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (support_set, query_set) in enumerate(train_loader):
        optimizer.zero_grad()

        # Move data to device (e.g., GPU)
        support_set = support_set.to(device)
        query_set = query_set.to(device)

        # Forward pass and backward pass
        loss = meta_learner(support_set, query_set)
        loss.backward()
        optimizer.step()

        # Log loss to Comet ML
        experiment.log_metric("loss", loss.item(), step=batch_idx + epoch * len(train_loader))

experiment.log_metric("final_loss", loss.item())
experiment.end()