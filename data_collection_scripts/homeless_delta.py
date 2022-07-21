import pandas as pd

# load data
_2015 = pd.read_excel("../data/raw/2015-Housing-Inventory-Count-Raw-File.xlsx", keep_default_na=False)
_2016 = pd.read_excel("../data/raw/2016-Housing-Inventory-Count-Raw-File.xlsx", keep_default_na=False)
_2017 = pd.read_excel("../data/raw/2017-Housing-Inventory-Count-Raw-File.xlsx", keep_default_na=False)
_2018 = pd.read_excel("../data/raw/2018-Housing-Inventory-County-RawFile.xlsx", keep_default_na=False)
_2019 = pd.read_excel("../data/raw/2019-Housing-Inventory-County-RawFile.xlsx", keep_default_na=False)
_2020 = pd.read_excel("../data/raw/2020-HIC-Raw-File.xlsx", keep_default_na=False)
_2021 = pd.read_csv("../data/raw/2021-HIC-Counts-by-State.csv", keep_default_na=False)

# remove duplicates
_2015 = _2015.drop_duplicates(subset=['Organization Name'])
print("2015", _2015.__len__())

_2016 = _2016.drop_duplicates(subset=['Organization Name'])
print("2016", _2016.__len__())

_2017 = _2017.drop_duplicates(subset=['Organization Name'])
print("2017", _2017.__len__())

_2018 = _2018.drop_duplicates(subset=['Organization Name'])
print("2018", _2018.__len__())

_2019 = _2019.drop_duplicates(subset=['Organization Name'])
print("2019", _2019.__len__())

_2020 = _2020.drop_duplicates(subset=['Organization Name'])
print("2020", _2020.__len__())

_2021 = _2021.drop_duplicates(subset=['Organization Name'])
print("2021", _2021.__len__())
