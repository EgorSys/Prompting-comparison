# %% [markdown]
# Load data

# %%
import pandas as pd

# %%
df = pd.read_csv('processed_data/test_prompting_df.csv')

# %% [markdown]
# Loading model

# %%
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import pipeline


# %%
feature_extractor = pipeline("feature-extraction", framework="pt", model='llama2_model/llama2_model', device=0)
def get_embedding(text):
    return feature_extractor(text,return_tensors = "pt")[0].numpy().mean(axis=0)

# %% [markdown]
# Getting embeddings (test df)

# %%
def data():
    for idx, row in df.iterrows():
        yield row['prompt']


embeddings = []
for features in feature_extractor(data(), return_tensors = "pt"):
    embedding = features[0].numpy().mean(axis=0)
    embeddings.append(embedding)
    
df['vector'] = embeddings

df.to_pickle('processed_data/prompts_test_embedding.pkl')


