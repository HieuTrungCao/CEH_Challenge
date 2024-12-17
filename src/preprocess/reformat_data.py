import pandas as pd

path = "data/GereralAI/train/question.csv"

data = pd.read_csv(path)

new_data = {}
new_data['index'] = data["index"]
new_data['summary_llmanswer'] = data["summary_llmanswer"]
new_data['ground_truth'] = data["ground_truth"]
new_data['is_correct'] = data["is_correct"]
new_data["Reason_1"] = data["Reason_1"]
new_data["Reason_2"] = data["Reason_2"]
new_data["Reason_3"] = data["Reason_3"]

data_columns = ['question', 'llm_answer']

key_word = ["A.", "B.", "C.", "D."]
new_questions = []
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

des_path = "data/GereralAI/train/reformat_question.csv"
new_df = pd.DataFrame(new_data)
new_df.to_csv(des_path, index=False)