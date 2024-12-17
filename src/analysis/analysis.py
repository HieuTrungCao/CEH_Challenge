import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv("data/GereralAI/train/question.csv")

title = 'Distribution of question + llm_answer Lengths'
data['question_llm_answer_length'] = data['question'].apply(len) + data['llm_answer'].apply(len)
plt.figure(figsize=(10, 5))
sns.histplot(data['question_llm_answer_length'], bins=50, kde=True)
plt.title(title)
plt.xlabel('Length of question')
plt.ylabel('Frequency')
plt.savefig(os.path.join("src/analysis/figs", title + '.png'))
plt.show()