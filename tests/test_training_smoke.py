import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from config import ModelConfig, TrainConfig
from dataset import get_dataloaders
from model import GPT
from train import make_lr_scheduler, make_optimizer, save_checkpoint


def run_training_smoke(directory: Path):
    tokens = np.arange(512, dtype=np.uint32) % 97
    train_path = directory / "train.bin"
    val_path = directory / "val.bin"
    tokens.tofile(train_path)
    tokens[:256].tofile(val_path)
    model_config = ModelConfig(
        vocab_size=97,
        max_seq_len=16,
        dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
        ffn_multiple_of=16,
    )
    train_config = TrainConfig(
        train_tokens=str(train_path),
        val_tokens=str(val_path),
        output_dir=str(directory / "run"),
        batch_size=2,
        gradient_accumulation_steps=1,
        total_steps=2,
        warmup_steps=1,
        eval_batches=1,
    )
    loader, _ = get_dataloaders(train_path, val_path, 16, 2)
    model = GPT(model_config)
    optimizer = make_optimizer(model, train_config)
    scheduler = make_lr_scheduler(optimizer, train_config)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    batch = next(iter(loader))
    output = model(batch[:, :-1])
    loss = F.cross_entropy(output.logits.reshape(-1, 97), batch[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    checkpoint_path = directory / "smoke.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        0,
        model_config,
        train_config,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert state["step"] == 0
    assert state["model_config"]["n_kv_heads"] == 2
    assert torch.isfinite(loss)


def test_training_smoke(tmp_path):
    run_training_smoke(tmp_path)
