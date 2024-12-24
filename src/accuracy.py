import argparse
import pandas as pd

if __name__ == "__main__":
    parses = argparse.ArgumentParser()
    parses.add_argument("-g", "--ground_truth", help="Ground truth file")
    parses.add_argument("-p", "--predict", help="Prediction file")
    args = parses.parse_args()

    ground_truth = pd.read_csv(args.ground_truth, encoding= 'unicode_escape')
    ground_truth = list(ground_truth["ground_truth"])

    preds = pd.read_csv(args.predict)
    preds = list(preds)
    for item in preds:
        item = item.replace("\n", "")
        item = item.lower()
        print("Item: ", item)
        item = item.split("the correct answer is ")[1].split()[0].split(".")[0]
        preds.append(item)

    count = 0
    for p, g in zip(preds, ground_truth):
        if p == g.lower():
            count += 1

    print("Accuracy: ", count / len(preds))