import os
import math
import argparse
import json
import time
import torch
from torch.cuda.amp import GradScaler
from torch import autocast
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models import DHVNClassification
from data import get_dataloaders
from utils import anneal_temperature, compute_gate_entropy_loss, is_distributed
from utils import compute_gate_usage, accumulate_gates, reduce_mean_gates_across_processes

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--base_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--per_gpu_batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--T0", type=int, default=10)
    p.add_argument("--Tmul", type=int, default=2)
    p.add_argument("--init_temp", type=float, default=10.0)
    p.add_argument("--min_temp", type=float, default=0.5)
    p.add_argument("--entropy_clip", type=float, default=1e-8)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--output_dir", type=str, default="./outputs")

    return p.parse_args()

def setup_distrib():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    return local_rank, rank, world_size

def train_one_epoch(model, epoch, sampler, loader, optimizer, scheduler, criterion, scaler, distributed, device, args):
    model.train()
    if distributed and sampler is not None:
        sampler.set_epoch(epoch)

    # metrics vars
    total_loss, total_cls, total_ent = 0.0, 0.0, 0.0
    correct, total_samples = 0, 0
    # Gating vars
    all_gates_sum = {}
    

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            logits, down_gates, up_gates = model(images, get_gates=True)
            cls_loss = criterion(logits, labels)

            gates = {**down_gates, **up_gates}

            tau = anneal_temperature(model, epoch, {"init_temp":args.init_temp, "min_temp":args.min_temp, "epochs":args.epochs})
            lambda_entropy = 0.1 * (tau / 10.0)
            ent_loss = compute_gate_entropy_loss(gates, lambda_entropy, eps=1e-8)
            loss = cls_loss + ent_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        # per-Iteration LR update (Fractional stepping)
        if scheduler is not None:
            scheduler.step(epoch + step / len(loader))
    
        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_ent += ent_loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

        # Collecting soft and hard_gates
        all_gates_sum = accumulate_gates(gates, all_gates_sum)

    # Calculating the mean gates usage for each epoch
    mean_gates = {}
    for k, v in all_gates_sum.items():
        mean_soft = (v["soft_sum"] / v["count"]).tolist()
        hard_frac = (v["hard_sum"] / v["count"]).tolist()
        mean_gates[k] = {"soft_mean": mean_soft, "hard_frac": hard_frac}

    train_loss = total_loss / len(loader)
    train_acc = 100 * correct / total_samples

    global_mean_gates = reduce_mean_gates_across_processes(mean_gates, total_samples)

    return train_loss, train_acc, global_mean_gates

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, save_path):
    '''
    Runs validation on the validation loader and returns top1, top5 accuracy
    '''
    model.eval()

    # metrics vars
    total_loss = 0.0
    total_samples, correct_top1, correct_top5 = 0, 0, 0
    # gating vars
    all_gates_sum = {}

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(device_type=device.type):
            logits, down_gates, up_gates = model(images, get_gates=True)
            loss = criterion(logits, labels)

        gates = {**down_gates, **up_gates}
        
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        # Top-k accuracy
        maxk = min(5, logits.size(1))
        _, pred = logits.topk(maxk, 1, True, True)
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))

        correct_top1 += correct[:, :1].sum().item()
        correct_top5 += correct[:, :maxk].sum().item()

        # collecting gates
        all_gates_sum = accumulate_gates(gates, all_gates_sum)
    
    #Computing the mean gate usage
    mean_gates = {}
    for k, v in all_gates_sum.items():
        mean_gates[k] = {
            "soft_mean": (v["soft_sum"] / v["count"]).tolist(),
            "hard_frac": (v["hard_sum"] / v["count"]).tolist()
        }

    # Reducing across all GPUs
    total_loss_tensor = torch.tensor([total_loss, correct_top1, correct_top5, total_samples], device=device)
    if is_distributed():
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    total_loss, correct_top1, correct_top5, total_samples = total_loss_tensor.tolist()
    
    avg_loss = total_loss / total_samples
    top1 = 100.0 * correct_top1 / total_samples
    top5 = 100.0 * correct_top5 / total_samples

    global_mean_gates = reduce_mean_gates_across_processes(mean_gates, total_samples)
    
    return avg_loss, top1, top5, global_mean_gates


def main():
    args = get_args()
    local_rank, rank, world_size = setup_distrib()
    torch.cuda.set_device(local_rank)

    os.makedirs(args.output_dir, exist_ok=True)

    # device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # dataloaders
    data_path, batch_size, num_workers = args.data_path, args.per_gpu_batch, args.num_workers
    distributed = ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1)
    train_loader, val_loader, train_sampler, val_sampler = get_dataloaders(data_path, batch_size=batch_size, distributed=distributed, num_workers=num_workers, world_size=world_size, rank=rank)

    # Building the Model
    model = DHVNClassification(in_ch=3, base_dim=args.base_dim, norm="group", num_classes=10).to(device)

    # Making sure of parameters contiguous
    '''
    for p in model.parameters():
        if not p.is_contiguous():
            p.data = p.data.contiguous()
    '''

    # Gating Log Var allocation
    gates_log_path = os.path.join(args.output_dir, "gates_log.json")
    all_gate_log = []

    if torch.cuda.device_count() > 1:
        device_ids = [local_rank] if torch.cuda.is_available() else None
        model = DDP(model, device_ids=device_ids, output_device=local_rank, find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T0, T_mult=args.Tmul, eta_min=args.min_lr)
    
    scaler = GradScaler(enabled=torch.cuda.is_available())


    for epoch in range(args.epochs):
        
        train_time_start = time.time()
        train_loss, train_acc, train_mean_gates = train_one_epoch(model, epoch, train_sampler, train_loader, optimizer, scheduler, criterion, scaler, distributed, device, args)
        
        # scheduler.step()    
        train_end_time = time.time()

        val_loss, top1, top5 = 0, 0, 0
        val_mean_gates = {}
        if val_loader is not None:
            val_loss, top1, top5, val_mean_gates = validate(
                model, val_loader, criterion, device,
                epoch=epoch+1, save_path=args.output_dir
            )
        val_end_time = time.time()

        # Checkpointing & Logging on Rank 0
        if (not dist.is_initialized()) or (dist.get_rank() == 0):
            current_lr = optimizer.param_groups[0]["lr"]
            # Logging 
            print(f"\n[Epoch {epoch+1}/{args.epochs}] (T:{train_end_time-train_time_start:.2f})s (V:{val_end_time-train_end_time:.2f})s "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                  f"| Val Loss: {val_loss:.4f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% "
                  f"| LR: {current_lr:.6f}")       
            
            print(f"train_Gate usage: {json.dumps(train_mean_gates, indent=2)}")

            # Saving Checkpoint
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            ck_pth = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Saving checkpoint to: {ck_pth}\n", end="-"*60)
            torch.save({
                "epoch": epoch+1,
                "model_state": state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_top1": top1,
                "val_top5": top5,
                "val_loss": val_loss,
            }, ck_pth)

            # storing Gate Logs
            gate_log = {
                "epoch": epoch+1,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_top1": top1,
                "val_top5": top5,
                "train_gates": train_mean_gates,
                "val_gates": val_mean_gates if val_loader is not None else {}
            }
            all_gate_log.append(gate_log)

            # updating th file epoch wise (jic)
            with open(gates_log_path, "w") as f:
                json.dump(all_gate_log, f, indent=2)
        if dist.is_initialized():
            dist.barrier()
    

    if dist.is_initialized():
        dist.destroy_process_group()
        
if __name__ == "__main__":
    main()    