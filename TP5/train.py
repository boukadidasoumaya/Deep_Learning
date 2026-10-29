
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import train_loader, test_loader
from text_utils import (
    pretrained_embeddings,
    word2idx,
    idx2word,
    tokenize,
    untokenize
)
from model import ImageCaptioningModel, resnet_pretrained


# --- Initialisation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImageCaptioningModel(
    cnn_model=resnet_pretrained,
    embedding_model=pretrained_embeddings,
    lstm_hidden_dim=512
).to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

criterion = nn.CrossEntropyLoss(ignore_index=0)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

writer = SummaryWriter("runs/captioning")

os.makedirs("checkpoints", exist_ok=True)

num_epochs = 15

print("Device :", device)
print("Nombre de batchs Train :", len(train_loader))
print("Nombre de batchs Test :", len(test_loader))


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    running_loss = 0

    current_lr = optimizer.param_groups[0]["lr"]

    # --- Entraînement ---
    for i, (images, captions) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device)

        # Tokenisation manuelle du batch (simplifiée)
        caps = torch.LongTensor(
            [tokenize(x, word2idx) for x in captions[0]]
        ).to(device)

        # On fournit toutes les séquences sauf le dernier mot en entrée
        outputs = model(images, caps[:, :-1])

        # On s'attend à prédire la séquence décalée d'un cran (captions[:, 1:])
        # Reshape pour la Loss : (batch_size * seq_len, vocab_size)
        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            caps[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            writer.add_scalar(
                'Loss/Train_Step',
                running_loss / 10,
                epoch * len(train_loader) + i
            )
            running_loss = 0

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{i + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}",
                flush=True
            )

    avg_train_loss = total_loss / len(train_loader)

    # --- Évaluation et Génération ---
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, captions in test_loader:
            images = images.to(device)

            caps = torch.LongTensor(
                [tokenize(x, word2idx) for x in captions[0]]
            ).to(device)

            outputs = model(images, caps[:, :-1])

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                caps[:, 1:].reshape(-1)
            )

            val_loss += loss.item()

            # --- Visualisation d'une prédiction ---
            predictions = outputs[-1].argmax(dim=-1)

            pred_text = " ".join(
                untokenize(predictions.cpu().numpy(), idx2word)
            )

            true_text = " ".join(
                untokenize(caps[-1][1:].cpu().numpy(), idx2word)
            )

    avg_val_loss = val_loss / len(test_loader)

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss : {avg_train_loss:.4f}")
    print(f"Val Loss   : {avg_val_loss:.4f}")
    print(f"Prediction : {pred_text}")
    print(f"Target     : {true_text}")
    print(f"LR         : {current_lr:.6f}\n", flush=True)

    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

    # Sauvegarde du modèle après chaque époque
    torch.save(
        model.state_dict(),
        f"checkpoints/captioning_epoch_{epoch + 1}.pth"
    )

    scheduler.step()
    writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

writer.close()

print("Entraînement terminé !")
