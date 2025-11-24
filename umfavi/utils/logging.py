from typing import Any
import torch
from umfavi.types import FeedbackType
import numpy as np

def update_epoch_log_dict(epoch_log_dict: dict[str, tuple[int, Any]], metrics_dict: dict[str, Any], fb_type: FeedbackType):
    epoch_log_dict = dict(epoch_log_dict)
    for key, value in metrics_dict.items():
        if key in epoch_log_dict:
            prev_count, prev_val = epoch_log_dict[key]
            updated_vals = (prev_count + 1, prev_val + value)
            epoch_log_dict[key] = updated_vals
            epoch_log_dict[f"{key}_{fb_type.value}"] = updated_vals
        else:
            initial_value = (1, value)
            epoch_log_dict[key] = initial_value
            epoch_log_dict[f"{key}_{fb_type.value}"] = initial_value
    return epoch_log_dict


def epoch_log_dict_to_wandb(epoch_log_dict: dict[str, tuple[int, Any]]):
    wandb_dict = {}
    for key, (count, agg_value) in epoch_log_dict.items():
        if count > 0:
            wandb_dict[f"epoch/{key}"] = agg_value / count
        else:
            wandb_dict[f"epoch/{key}"] = np.nan
    return wandb_dict

def console_log_eval_metrics(eval_metrics: dict[str, Any]):
    print(f"  Evaluation Results:")
    if "eval/epic_distance" in eval_metrics:
        print(f"    EPIC Distance: {eval_metrics['eval/epic_distance']:.6f}")
    if "eval/regret" in eval_metrics:
        print(f"    Regret: {eval_metrics['eval/regret']:.6f}")
    if "eval/negative_log_likelihood" in eval_metrics:
        print(f"    Eval NLL: {eval_metrics['eval/negative_log_likelihood']:.4f}")
    if "eval/kl_divergence" in eval_metrics:
        print(f"    Eval KL: {eval_metrics['eval/kl_divergence']:.4f}")
    if "eval/td_error" in eval_metrics:
        print(f"    Eval TD Error: {eval_metrics['eval/td_error']:.4f}")
    print()


def console_log_batch_metrics(aggregated_loss_dict: dict[str, Any], step: int, steps_per_epoch: int, total_loss: torch.Tensor):
    # Console logging
    nll = aggregated_loss_dict.get("negative_log_likelihood", 0.0)
    kl = aggregated_loss_dict.get("kl_divergence", 0.0)
    td = aggregated_loss_dict.get("td_error", 0.0)
    print(f"  Step {step}/{steps_per_epoch-1} | Loss: {total_loss.item():.4f} | NLL: {nll:.4f} | KL: {kl:.4f} | TD: {td:.4f}")