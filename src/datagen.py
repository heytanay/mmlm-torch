import nltk
import torch
import warnings
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from pytorch_pretrained_biggan.utils import IMAGENET
from transformers import is_tf_available, AutoTokenizer, AutoModel
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, one_hot_from_names

nltk.download('wordnet')
warnings.simplefilter('ignore')

from .config import Config
from .datagen_functions import *

if __name__ == "__main__":
    # Get the class names
    img_embeddings, text_embeddings = [], []
    words_dataset = get_simplified_classnames()
    examples_dataset = generate_sentences(words_dataset)
    
    assert len(words_dataset) == len(examples_dataset), "Lengths of the classes and corresponding sentences must be same"

    classes = words_dataset
    all_sentences = examples_dataset

    # Iterate over every pair of class and corresponding sentence
    for cls, sent in tqdm(zip(classes, all_sentences), total=len(classes)):
        # Debug print("Class: {}, Sentence: {}".format(cls, sent))
        try:
            # Image embedding function
            img_emb = generate_image_embeddings(Config, cls).view(128)
            
            # Text embedding function
            text_emb = generate_text_embeddings(Config, sent)
            
            img_embeddings.append(img_emb)
            text_embeddings.append(text_emb)
        except AssertionError:
            continue
    
    # Save the Image and text embeddings
    img_emb = [tensor.detach().numpy() for tensor in img_embeddings]
    text_emb = [tensor.detach().numpy() for tensor in text_embeddings]

    np.save("../working/image_embeddings_large", img_emb)
    np.save("../working/text_embeddings_large", text_emb)