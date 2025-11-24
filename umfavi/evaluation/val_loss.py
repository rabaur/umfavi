from torch.utils.data import DataLoader
from umfavi.types import FeedbackType
from umfavi.multi_fb_model import MultiFeedbackTypeModel
from umfavi.evaluation.get_batch import get_batch

def compute_eval_loss(val_dataloaders: dict[FeedbackType, DataLoader], active_feedback_types: list[FeedbackType], multi_fb_model: MultiFeedbackTypeModel):
    assert not multi_fb_model.training, "Model is not in evaluation mode"
    dloader_iters = {fb_type: iter(val_dataloaders[fb_type]) for fb_type in active_feedback_types}
    eval_loss_dict = {}
    for fb_type in active_feedback_types:
        batch = get_batch(dloader_iters, val_dataloaders, fb_type)
        loss_dict = multi_fb_model(**batch)

        # Log total loss metrics
        for k, v in loss_dict.items():
            if k not in eval_loss_dict:
                eval_loss_dict[k] = (0, 0.0)
            count, agg_val = eval_loss_dict[k]
            eval_loss_dict[k] = (count + 1, agg_val + v)
        
        # Log feedback-specific metrics
        for k, v in loss_dict.items():
            key = f"{k}_{fb_type.value}"
            if key not in eval_loss_dict:
                eval_loss_dict[key] = (0, 0.0)
            count, agg_val = eval_loss_dict[key]
            eval_loss_dict[key] = (count + 1, agg_val + v)
    
    # Average loss metrics
    final_dict = {}
    for k, (count, agg_val) in eval_loss_dict.items():
        if count > 0:
            final_dict[f"eval/{k}"] = agg_val / count
    return final_dict