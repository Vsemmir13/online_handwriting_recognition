import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, Optional
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.dont_write_bytecode = True

from MODEL.model.ptohwr import PTOHWR
from MODEL.processing_dataset.base_classes import PointSet
from MODEL.constants import DATA, SPLIT, PREPROCESS
from MODEL.processing_dataset.generator import IAMDataset, make_iam_collate_fn
from MODEL.decoding.decoding_converter import label2txt
from MODEL.model.metrics import character_error_rate, word_error_rate
from MODEL.processing_dataset.base_classes import Point, Line


class PTOHWRWorker:
    def __init__(
        self,
        model: PTOHWR,
        device: str = "mps" if torch.backends.mps.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.9
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        print(f"Model initialized on device: {device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def get_preprocess_scheme(self, scheme_id=1):
        scheme_attr = f"SCHEME{scheme_id}"
        if hasattr(PREPROCESS, scheme_attr):
            return getattr(PREPROCESS, scheme_attr)
        else:
            return PREPROCESS.CURRENT_SCHEME

    def create_dataloaders(self, batch_size, split, preprocess, npz, preload):
        train_dataset = IAMDataset(
            split=SPLIT.TRAIN,
            preprocess=preprocess,
            npz=npz,
            preload=preload,
            inout_ratio=4,
        )
        val_dataset = IAMDataset(
            split=SPLIT.VAL1,
            preprocess=preprocess,
            npz=npz,
            preload=preload,
            inout_ratio=4,
        )
        train_collate = make_iam_collate_fn(inout_ratio=train_dataset.inout_ratio)
        val_collate = make_iam_collate_fn(inout_ratio=val_dataset.inout_ratio)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_collate
        )
        return train_loader, val_loader

    def process_iam_batch(
        self, batch: Dict[str, torch.Tensor], preprocess_scheme: dict = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = batch["inputs"]
        targets = batch["labels_padded"]
        _ = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        batch_size, seq_len, feat_dim = inputs.shape
        trajectory = inputs[:, :, :3]
        trajectory_coords = inputs[:, :, :2]
        image = self._trajectory_to_image(trajectory, trajectory_coords, preprocess_scheme)
        target_chars = self._process_targets(targets, target_lengths)
        return trajectory, image, trajectory_coords, target_chars

    def _trajectory_to_image(
        self, trajectory: torch.Tensor, coords: torch.Tensor, preprocess_scheme: dict = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = trajectory.shape
        if preprocess_scheme is None:
            preprocess_scheme = PREPROCESS.CURRENT_SCHEME
        x_coords = coords[:, :, 0]
        y_coords = coords[:, :, 1]
        x_min = float(x_coords.min().item())
        x_max = float(x_coords.max().item())
        y_min = float(y_coords.min().item())
        y_max = float(y_coords.max().item())
        img_height = 32
        img_width = max(64, int(seq_len // 2))
        traj_np = trajectory.detach().cpu().numpy()
        coords_np = coords.detach().cpu().numpy()
        images = []
        for b in range(batch_size):
            canvas = np.zeros((img_height, img_width), dtype=np.float32)
            points = []
            for t in range(seq_len):
                pen_down = float(traj_np[b, t, 2])
                if pen_down > 0:
                    x = float(coords_np[b, t, 0])
                    y = float(coords_np[b, t, 1])
                    points.append(Point(1, 0, x, y))
            if len(points) < 2:
                images.append(canvas)
                continue
            pointset = PointSet(points=points, w=x_max-x_min, h=y_max-y_min)
            pointset.preprocess(**preprocess_scheme)
            for t in range(len(pointset.points) - 1):
                p1 = pointset.points[t]
                p2 = pointset.points[t + 1]
                line = Line(p1, p2, eos=(t == len(pointset.points) - 2))
                length = max(1.0, line.length())
                steps = max(2, int(length))
                for s in range(steps):
                    pc = s / (steps - 1)
                    pt = line.interpolate(pc)
                    xn = (pt.x - x_min) / (x_max - x_min + 1e-8)
                    yn = (pt.y - y_min) / (y_max - y_min + 1e-8)
                    xi = int(xn * (img_width - 1))
                    yi = int(yn * (img_height - 1))
                    if 0 <= xi < img_width and 0 <= yi < img_height:
                        canvas[yi, xi] = 1.0
            images.append(canvas)
        images = np.stack(images, axis=0).astype(np.float32)
        return torch.from_numpy(images).unsqueeze(1)

    def _process_targets(
        self, targets: torch.Tensor, target_lengths: torch.Tensor
    ) -> torch.Tensor:
        batch_size = targets.shape[0]
        max_len = targets.shape[1]
        processed_targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
        for b in range(batch_size):
            length = target_lengths[b]
            if length > 0:
                processed_targets[b, :length] = targets[b, :length]
        return processed_targets

    def train_epoch(self, dataloader: DataLoader, epoch) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_loss_1d = 0.0
        total_loss_2d = 0.0
        total_loss_align = 0.0
        num_batches = 0
        pbar = tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}"
        )
        for batch_idx, batch in pbar:
            preprocess_scheme = self.get_preprocess_scheme(1)
            trajectory, image, trajectory_coords, target_chars = self.process_iam_batch(
                batch, preprocess_scheme
            )
            trajectory = trajectory.to(self.device)
            image = image.to(self.device)
            trajectory_coords = trajectory_coords.to(self.device)
            target_chars = target_chars.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(
                trajectory=trajectory,
                image=image,
                trajectory_coords=trajectory_coords,
                target_chars=target_chars,
                training=True,
            )
            loss = outputs["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total_loss_1d += outputs["loss_1d"].item()
            total_loss_2d += outputs["loss_2d"].item()
            total_loss_align += outputs["loss_align"].item()
            num_batches += 1
            pbar.set_postfix(
                {
                    "Loss": f"{total_loss / num_batches:.4f}",
                    "1D": f"{total_loss_1d / num_batches:.4f}",
                    "2D": f"{total_loss_2d / num_batches:.4f}",
                    "Align": f"{total_loss_align / num_batches:.4f}",
                    "LR": f"{self.optimizer.param_groups[0]['lr']:.1e}",
                }
            )
        self.scheduler.step()
        return {
            "total_loss": total_loss / num_batches,
            "loss_1d": total_loss_1d / num_batches,
            "loss_2d": total_loss_2d / num_batches,
            "loss_align": total_loss_align / num_batches,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                preprocess_scheme = self.get_preprocess_scheme(1)
                trajectory, image, trajectory_coords, target_chars = (
                    self.process_iam_batch(batch, preprocess_scheme)
                )
                trajectory = trajectory.to(self.device)
                image = image.to(self.device)
                trajectory_coords = trajectory_coords.to(self.device)
                target_chars = target_chars.to(self.device)
                outputs = self.model(
                    trajectory=trajectory,
                    image=image,
                    trajectory_coords=trajectory_coords,
                    target_chars=target_chars,
                    training=True,
                )
                total_loss += outputs["total_loss"].item()
                num_batches += 1
        return {"val_loss": total_loss / num_batches}

    @torch.no_grad()
    def infer_dataset(
        self,
        split: str = SPLIT.TEST,
        batch_size: int = 32,
        preprocess: int = 6,
        npz: bool = True,
        preload: bool = True,
        max_len: int = 128,
    ) -> Dict[str, Any]:
        self.model.eval()
        dataset = IAMDataset(
            split=split,
            preprocess=preprocess,
            npz=npz,
            preload=preload,
            inout_ratio=4,
            pred=False,
        )
        collate = make_iam_collate_fn(inout_ratio=dataset.inout_ratio)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        y_true: list = []
        y_pred: list = []
        for batch in tqdm(dataloader, desc=f"Infer {split}"):
            preprocess_scheme = self.get_preprocess_scheme(preprocess)
            trajectory, image, trajectory_coords, target_chars = self.process_iam_batch(
                batch, preprocess_scheme
            )
            trajectory = trajectory.to(self.device)
            trajectory_coords = trajectory_coords.to(self.device)
            encoded = self.model.inference(trajectory, trajectory_coords)
            ypred_lengths = batch["ypred_lengths"].to(self.device)
            max_out_len = int(ypred_lengths.max().item())
            pred_ids_all = self.model.decode_iterative(
                encoded, 
                max_length=max_out_len, 
                blank_idx=DATA.BLANK_IDX
            )
            for i in range(pred_ids_all.size(0)):
                L = int(ypred_lengths[i].item())
                ids = pred_ids_all[i, :L].tolist()
                y_pred.append(label2txt(ids, remove_dup=False))
            labels_padded = batch["labels_padded"]
            target_lengths = batch["target_lengths"]
            for i in range(labels_padded.size(0)):
                L = int(target_lengths[i].item())
                if L > 0:
                    ids = labels_padded[i, :L].tolist()
                else:
                    ids = []
                y_true.append(label2txt(ids, remove_dup=False))
        cer_corpus = character_error_rate(y_true, y_pred, corpus_level=True)
        wer_corpus = word_error_rate(y_true, y_pred, corpus_level=True)
        return {
            "num_samples": len(y_true),
            "cer": cer_corpus,
            "wer": wer_corpus,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filepath: str):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Checkpoint loaded from {filepath}")
            return checkpoint["epoch"], checkpoint["metrics"]
        else:
            print(f"No checkpoint found at {filepath}")
            return 0, {}

    def train(
        self,
        num_epochs: int = 50,
        batch_size: int = 32,
        save_interval: int = 5,
        checkpoint_dir: str = "checkpoints",
        split: str = SPLIT.TRAIN,
        preprocess: int = 6,
        npz: bool = True,
        preload: bool = True,
    ):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("Creating IAM dataloaders...")
        train_loader, val_loader = self.create_dataloaders(
            batch_size=batch_size,
            split=split,
            preprocess=preprocess,
            npz=npz,
            preload=preload,
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Learning Rate: {train_metrics['learning_rate']:.6f}")
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(
                    epoch,
                    {**train_metrics, **val_metrics},
                    os.path.join(checkpoint_dir, "best_model.pth"),
                )
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")


def launch_training(config):
    print("Launching training...")
    print("Creating  PTOHWR model with vocab size:", len(DATA.CHARS))
    model = PTOHWR(
        vocab_size=len(DATA.CHARS),
        feature_dim=config["model"]["feature_dim"],
        num_conv1d_layers=config["model"]["num_conv1d_layers"],
        num_conv2d_layers=config["model"]["num_conv2d_layers"],
        num_bigru_layers=config["model"]["num_bigru_layers"],
        num_transformer_layers=config["model"]["num_transformer_layers"],
        alignment_weight=config["model"]["alignment_weight"],
        dropout=config["model"]["dropout"],
    )
    print("Model created successfully!")
    print("Creating worker...")
    trainer = PTOHWRWorker(
        model=model,
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    print("Worker created successfully!")
    trainer.train(
        num_epochs=config["train"]["epochs"],
        batch_size=config["train"]["batch_size"],
        save_interval=config["train"]["save_interval"],
        preprocess=config["train"]["preprocess"],
        checkpoint_dir=config["train"]["checkpoint_dir"],
    )


@torch.no_grad()
def launch_inference(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: Optional[str] = "mps" if torch.backends.mps.is_available() else "cpu",
    split: str = SPLIT.TEST,
):
    print("Building model for inference...")
    model = PTOHWR(
        vocab_size=len(DATA.CHARS),
        feature_dim=config["model"]["feature_dim"],
        num_conv1d_layers=config["model"]["num_conv1d_layers"],
        num_conv2d_layers=config["model"]["num_conv2d_layers"],
        num_bigru_layers=config["model"]["num_bigru_layers"],
        num_transformer_layers=config["model"]["num_transformer_layers"],
        alignment_weight=config["model"]["alignment_weight"],
        dropout=config["model"]["dropout"],
    )
    worker = PTOHWRWorker(model=model)
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state)
    worker.model.load_state_dict(state_dict)
    worker.model.eval()
    print("Running inference over dataset...")
    metrics = worker.infer_dataset(
        split=split,
        batch_size=config["train"]["batch_size"],
        preprocess=config["train"]["preprocess"],
        npz=True,
        preload=True,
    )
    print(f"Samples: {metrics['num_samples']}")
    print(f"CER: {metrics['cer']:.4f}")
    print(f"WER: {metrics['wer']:.4f}")
    return metrics


@torch.no_grad()
def launch_example(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: Optional[str] = "mps" if torch.backends.mps.is_available() else "cpu",
    split: str = SPLIT.TEST,
    idx: int = 0,
):
    print("Building model for single-sample example...")
    model = PTOHWR(
        vocab_size=len(DATA.CHARS),
        feature_dim=config["model"]["feature_dim"],
        num_conv1d_layers=config["model"]["num_conv1d_layers"],
        num_conv2d_layers=config["model"]["num_conv2d_layers"],
        num_bigru_layers=config["model"]["num_bigru_layers"],
        num_transformer_layers=config["model"]["num_transformer_layers"],
        alignment_weight=config["model"]["alignment_weight"],
        dropout=config["model"]["dropout"],
    )
    worker = PTOHWRWorker(model=model, device=device)
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state)
    worker.model.load_state_dict(state_dict)
    worker.model.eval()
    print("Preparing dataset and sample...")
    ds = IAMDataset(
        split=split,
        preprocess=config["train"]["preprocess"],
        npz=True,
        preload=True,
        inout_ratio=4,
    )
    if idx < 0 or idx >= len(ds):
        print(f"Index {idx} out of range, using 0")
        idx = 0
    x, labels, x_len, y_len, sample = ds[idx]
    collate = make_iam_collate_fn(inout_ratio=ds.inout_ratio)
    batch = collate([(x, labels, x_len, y_len, sample)])
    preprocess_scheme = worker.get_preprocess_scheme(config["train"]["preprocess"])
    trajectory, image, trajectory_coords, target_chars = worker.process_iam_batch(
        batch, preprocess_scheme
    )
    trajectory = trajectory.to(worker.device)
    trajectory_coords = trajectory_coords.to(worker.device)
    encoded = worker.model.inference(trajectory, trajectory_coords)
    ypred_lengths = batch["ypred_lengths"].to(worker.device)
    max_out_len = int(ypred_lengths.max().item())
    pred_ids_all = worker.model.decode_iterative(
        encoded, 
        max_length=max_out_len, 
        blank_idx=DATA.BLANK_IDX
    )
    actual_length = min(ypred_lengths[0], pred_ids_all.shape[1])
    pred_ids = pred_ids_all[0, :actual_length].tolist()
    pred_txt = label2txt(pred_ids, remove_dup=False)
    gt_txt = label2txt(labels[:y_len].tolist(), remove_dup=False)
    print(f"Sample: {sample.name}")
    print(f"GT   : {gt_txt}")
    print(f"Pred : {pred_txt}")
    print("Displaying trajectory image...")
    img_np = image[0, 0].detach().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(img_np, cmap='gray', aspect='auto')
    plt.title(f"Sample: {sample.name}\nGT: {gt_txt}\nPred: {pred_txt}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return {"sample": sample.name, "gt": gt_txt, "pred": pred_txt}
