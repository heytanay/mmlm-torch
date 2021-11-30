from transformers import AutoModel, AutoTokenizer
from pytorch_pretrained_biggan import BigGAN

class Config:
    wandb = True
    MODEL_CLASS = AutoModel
    gan_model = BigGAN.from_pretrained('biggan-deep-128')
    lm_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    lm_model = MODEL_CLASS.from_pretrained('distilbert-base-uncased')
    PATTERNS = [
        'i saw a <WORD>.',
        'i love a <WORD>.',
        'i like a <WORD>.',
        'i found a <WORD>.',
        'i desire a <WORD>.',
        'i want a <WORD>.',
        'i hate a <WORD>.',
        'i use a <WORD>.',
        'i conjure a <WORD>.',
        'i need a <WORD>.'
    ]
    pat_len = len(PATTERNS)