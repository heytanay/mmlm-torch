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

from config import Config

nltk.download('wordnet')
warnings.simplefilter('ignore')

def generate_image_embeddings(config, class_name):
    """
    Generates Dense Class Vector Embeddings from the class name
    """
    class_vector = one_hot_from_names([class_name])
    class_vector = torch.from_numpy(class_vector)
    dense_class_vector = config.gan_model.embeddings(class_vector)

    return dense_class_vector

def generate_text_embeddings(config, sentence):
    """
    Gets the Language Model output after passing in the sentence
    """
    tokens = config.lm_tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        lm_output = config.lm_model(tokens)
        lm_hidden_states = lm_output[0]
        lm_hidden_states_first_example = lm_hidden_states[0]
    return lm_hidden_states_first_example

def get_simplified_classnames():
    """
    Returns simplified class names
    """
    class_to_synset = dict((v, wn.synset_from_pos_and_offset('n', k)) for k, v in IMAGENET.items())
    words_dataset = []
    all_words = set()
    for i, synset in tqdm(class_to_synset.items()):
        current_synset = synset
        while current_synset:
            for lemma in current_synset.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name in all_words:
                    continue
                if Config.lm_tokenizer.convert_tokens_to_ids(name) != Config.lm_tokenizer.unk_token_id:
                    for _ in range(Config.pat_len):
                        words_dataset.append(name)
                    all_words.add(name)
                    current_synset = False
                    break
            if current_synset and current_synset.hypernyms():
                current_synset = current_synset.hypernyms()[0]
            else:
                current_synset = False
    return words_dataset

def generate_sentences(words_dataset):
    """
    Generate sentences from the given pattern
    """
    examples_dataset = []
    words_dataset = [i for n, i in enumerate(words_dataset) if i not in words_dataset[:n]]

    for i, word in tqdm(enumerate(words_dataset)):
        current_patterns = [pat.replace('<WORD>', word) for pat in Config.PATTERNS]
        examples_dataset.extend(current_patterns)
    return examples_dataset

