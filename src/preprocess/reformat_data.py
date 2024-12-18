import pandas as pd

path_hieu = "data/GereralAI/train/question.csv"
path_kien = "data/GereralAI/train/kien.csv"

data_hieu = pd.read_csv(path_hieu, encoding="utf8")
data_kien = pd.read_csv(path_kien, encoding="utf8")

new_data = {}
new_data['summary_llmanswer'] = list(data_hieu["summary_llmanswer"]) + list(data_kien["summary_llmanswer"])
new_data['ground_truth'] = list(data_hieu["ground_truth"]) + list(data_kien["ground_truth"])
new_data["Reason_1"] = list(data_hieu["Reason_1"]) + list(data_kien["Reason_1"])
new_data["Reason_2"] = list(data_hieu["Reason_2"]) + list(data_kien["Reason_2"])
new_data["Reason_3"] = list(data_hieu["Reason_3"]) + list(data_kien["Reason_3"])

data_columns = ['question', 'llm_answer']

key_word = ["A.", "B.", "C.", "D."]
new_questions = []
for data in [data_hieu, data_kien]:
    for item in data[data_columns[0]]:
        item = item.replace("\nA.", "A.")
        item = item.replace("\nB.", "B.")
        item = item.replace("\nC.", "C.")
        item = item.replace("\nD.", "D.")
        item = item.replace("A.", "\nA.")
        item = item.replace("B.", "\nB.")
        item = item.replace("C.", "\nC.")
        new_item = item.replace("D.", "\nD.")
        new_questions.append(new_item)
new_data[data_columns[0]] = new_questions

new_answers = []
for data in [data_hieu, data_kien]:
    for item in data[data_columns[1]]:
        item = item.replace("\nA.", "A.")
        item = item.replace("\nB.", "B.")
        item = item.replace("\nC.", "C.")
        item = item.replace("\nD.", "D.")
        item = item.replace("A.", "A.\n")
        item = item.replace("B.", "B.\n")
        item = item.replace("C.", "C.\n")
        new_item = item.replace("D.", "D.\n")
        new_item = "The correct answer is " + new_item
        new_answers.append(new_item)
new_data[data_columns[1]] = new_answers


for k, item in new_data.items():
    print("{} len: {}".format(k, len(item)))
des_path = "data/GereralAI/train/train.csv"
new_df = pd.DataFrame(new_data)
new_df.to_csv(des_path, index=False)