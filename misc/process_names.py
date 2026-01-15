import pandas as pd
import zipfile
import io
import requests

url = "https://www.ssa.gov/oact/babynames/names.zip"
content = requests.get(url).content

with zipfile.ZipFile(io.BytesIO(content)) as z:
    files = z.namelist()
    df_list = []
    for file in files:
        if file.endswith(".txt"):
            year = int(file[3:7])
            temp = pd.read_csv(z.open(file), names=["name", "sex", "n"])
            temp["year"] = year
            df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)


names_to_check = ["Alex", "Taylor", "Emily", "Jordan"]

gender_stats = (
    df[df["name"].isin(names_to_check)]
    .groupby(["name", "sex"])["n"]
    .sum()
    .unstack(fill_value=0)
)

gender_stats["male_prop"] = gender_stats.get("M", 0) / gender_stats.sum(axis=1)
gender_stats["female_prop"] = gender_stats.get("F", 0) / gender_stats.sum(axis=1)

