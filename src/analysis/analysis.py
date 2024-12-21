import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def custom_len(text):
    return len(text.split())

data = pd.read_csv("data/GereralAI/train/train.csv")

title = 'Distribution of question + llm_answer Lengths on train'
data['question_llm_answer_length'] = data['question'].apply(custom_len) + data['llm_answer'].apply(custom_len)
plt.figure(figsize=(10, 5))
sns.histplot(data['question_llm_answer_length'], bins=50, kde=True)
plt.title(title)
plt.xlabel('Length of question')
plt.ylabel('Frequency')
plt.savefig(os.path.join("src/analysis/figs", title + '.png'))
plt.show()