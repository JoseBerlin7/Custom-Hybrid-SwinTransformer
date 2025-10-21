import math
import torch
import torch.distributed as dist

def is_distributed():
    return dist.is_available() and dist.is_initialized()
    
def cosine_tau_schedule(epoch, init_temp, min_temp, total_epochs):
    '''
    Smooth cosine annealing for τ (temperature).
    '''
    tau = min_temp + 0.5 * (init_temp - min_temp) * (1 + math.cos(math.pi * epoch / total_epochs))
    return tau

def anneal_temperature(model, epoch, cfg):
    '''
    Exponential annealing of Gumbel-softmax temperature along epochs
    '''
    tau = cosine_tau_schedule(epoch, cfg["init_temp"], cfg["min_temp"], cfg["epochs"])
    for module in model.modules():
        if module.__class__.__name__ == "GumbelGate":
            module.set_temperature(tau)            
    return tau

def compute_gate_entropy_loss(gates, lambda_entropy=0.1, eps=1e-8):
    '''
    Compute entropy regularization from all gates in the network
    '''
    entropy_loss = 0.0
    for gate in gates.values():
        # we add a small epsilon(eps) value to avoid log(0)
        probs = gate["soft"] + eps
        entropy = -(probs * torch.log(probs)).sum(dim=1).mean()
        uniform = torch.full_like(probs, 1.0 / probs.size(-1))
        kl = (probs * (torch.log(probs) - torch.log(uniform))).sum(dim=1).mean()
        entropy_loss += entropy + 0.1 * kl
    return -lambda_entropy * entropy_loss

def compute_gate_usage(gates):
    '''
    Calculates % of samples that took each path (Conv / Transformer / Fusion).
    '''
    usage_stats = {}
    for layer_name, gate in gates.items():
        hard = gate["hard"]  # shape (B, num_choices)
        counts = hard.sum(dim=0)
        usage = (counts / hard.shape[0]) * 100
        usage_stats[layer_name] = usage.detach().cpu().numpy().tolist()
    return usage_stats

def reduce_mean_gates_across_processes(local_mean_gates, total_samples):
    '''
    Reduces per-process mean gate usage (soft_mean & hard_frac) into a global mean across all ranks.
    Uses weighted averaging based on number of samples processed by each process.

    Args:
        local_mean_gates (dict): Per-rank mean gate dictionary of structure:
            { "gate_name": {"soft_mean": [...], "hard_frac": [...]} }
        total_samples (int): Total number of samples processed in this rank.

    Returns:
        dict: Global averaged gate usage on rank 0. None on other ranks.
    '''
    obj = {"mean_gates": local_mean_gates, "samples": total_samples}

    if dist.is_initialized():
        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, obj)

        if dist.get_rank() == 0:
            sums = {}
            counts = {}
            global_mean_gates = {}

            for entry in gathered:
                mg = entry["mean_gates"]
                n = entry["samples"]
                for k, stats in mg.items():
                    soft = torch.tensor(stats["soft_mean"])
                    hard = torch.tensor(stats["hard_frac"])
                    if k not in sums:
                        sums[k] = {
                            "soft_sum": torch.zeros_like(soft),
                            "hard_sum": torch.zeros_like(hard)
                        }
                        counts[k] = 0
                    sums[k]["soft_sum"] += soft * n
                    sums[k]["hard_sum"] += hard * n
                    counts[k] += n
            
            for k in sums:
                global_mean_gates[k] = {
                    "soft_mean": (sums[k]["soft_sum"] / counts[k]).tolist(),
                    "hard_frac": (sums[k]["hard_sum"] / counts[k]).tolist()
                }
        else:
            global_mean_gates = None
    else:
        global_mean_gates = local_mean_gates
    
    return global_mean_gates

def accumulate_gates(gate_dict, all_gates_sum):
    for k, v in gate_dict.items():
        soft = v["soft"].detach().cpu()
        hard = v["hard"].detach().cpu().float()
        bs = soft.shape[0]

        if k not in all_gates_sum:
            all_gates_sum[k] = {
                "soft_sum": torch.zeros_like(soft[0]),
                "hard_sum": torch.zeros_like(hard[0]),
                "count": 0
            }

        all_gates_sum[k]["soft_sum"] += soft.sum(dim=0)
        all_gates_sum[k]["hard_sum"] += hard.sum(dim=0)
        all_gates_sum[k]["count"] += bs
    
    return all_gates_sum







