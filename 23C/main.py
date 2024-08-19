import pandas as pd

path = "E:\\2023C题\\附件3.xlsx"
excel3 = pd.read_excel(path)
columns = list(excel3['日期'].drop_duplicates(keep='first'))
index = list(excel3['单品编码'].drop_duplicates(keep='first'))
new = pd.DataFrame(index=index, columns=columns)
for x in index:
    for y in columns:
        pass