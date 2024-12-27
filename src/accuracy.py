import argparse
import pandas as pd

if __name__ == "__main__":
    parses = argparse.ArgumentParser()
    parses.add_argument("-g", "--ground_truth", help="Ground truth file")
    parses.add_argument("-p", "--predict", help="Prediction file")
    args = parses.parse_args()

    ground_truth = pd.read_csv(args.ground_truth, encoding= 'unicode_escape')
    print("Baseline: ", sum(list(ground_truth["_is_correct"])) / len(list(ground_truth["_is_correct"])))
    ground_truth = list(ground_truth["ground_truth"])

    answers = pd.read_csv(args.predict)
    answers = list(answers["llm_answer"])
    preds = []
    for i, item in enumerate(answers):
        if "correct answer is" not in item:
            preds.append("l")
            continue
        item = item.replace("\n", "")
        item = item.lower()
        item = item.split("the correct answer is ")[1].split()[0].split(".")[0]
        preds.append(item)

    count = 0
    for p, g in zip(preds, ground_truth):
        print("pred: {}, ground_truth: {}".format(p, g))
        if p == g.lower():
            count += 1

    print("Accuracy: ", count / len(preds))