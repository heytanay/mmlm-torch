import wandb

def wandb_log(**kwargs):
    """
    Logs a key value pair to W&B
    """
    for key, value in kwargs.items():
        wandb.log({key: value})