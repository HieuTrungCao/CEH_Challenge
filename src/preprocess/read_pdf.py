import PyPDF2

path_file = "data/GereralAI/CEH_Certified_Ethical_Hacker_Bundle,_5th_Edition_Matt_Walker_2022-1.pdf"
reader = PyPDF2.PdfFileReader(path_file)

for i in range(663, 665):
    print(reader.pages[i].extract_text())

# print(reader.pages[635].extract_text())