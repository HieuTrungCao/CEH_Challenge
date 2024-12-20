from PyPDF2 import PdfReader
import pandas as pd

reader = PdfReader("data/GereralAI/Sybex_CEH_v10_Certified_Ethical.pdf")

text = []
for page in  reader.pages[40: ]:
    sentences = page.extract_text().replace("\n", " ").split(".")
    for sentence in sentences[1 : -1]:
        if len(sentence) > 100 and isinstance(sentence, str):
            text.append(sentence) 

data = {
    "sentence": text
}

data = pd.DataFrame(data)
data.to_csv("./data/GereralAI/train/my_data.csv", index=False)
# print(reader.getOutlines())