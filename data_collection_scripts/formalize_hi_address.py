import json
import pandas as pd
import googlemaps

api_key = "AIzaSyB3rZUfAkjl6d1B8wD5IDAVSP4B6RyjxGM"
maps = googlemaps.Client(key=api_key)
api_count = 0


# Formalize Address
def formalize_address(address, city):
    text = address + " " + city
    address = maps.places_autocomplete(input_text=text, types="address", components={"country": ["US"]})[0]
    return address["description"]


# by State
with open("../data/clean/state_names.json", "r") as file:
    states = json.loads(file.read())

# load housing inventory data
housing_inventory = pd.read_excel("data/raw/2020-HIC-Raw-File.xlsx", sheet_name="HIC_RawData2020",
                                  usecols=["Organization Name", "address1", "city", "state", "zip",
                                           "Total Beds"], na_filter=False)
housing_inventory = housing_inventory.drop_duplicates(subset=["address1"])

# shelter by state
housing_inventory_json = {}
for i in range(len(housing_inventory)):
    address = str.lower(str(housing_inventory.iloc[i]["address1"])).strip()

    if len(address) > 0 and housing_inventory.iloc[i]["Total Beds"] > 0:
        # Formalize Address
        city = str(housing_inventory.iloc[i]["city"]).strip()
        try:
            address = formalize_address(address, city)
            api_count += 1
            print("API REQ", api_count)
            key = address

            if key not in housing_inventory_json:
                housing_inventory_json[key] = {
                    "beds": str(housing_inventory.iloc[i]["Total Beds"]),
                    "state": housing_inventory.iloc[i]["state"],
                    "name": str(housing_inventory.iloc[i]["Organization Name"]).strip(),
                    "address": address,
                    "city": city,
                    "zip": str(housing_inventory.iloc[i]["zip"]).strip()
                }
            else:
                housing_inventory_json[key]["beds"] = str(int(housing_inventory_json[key]["beds"]) + housing_inventory.iloc[i]["Total Beds"])
        except Exception as error:
            print("Missed B", address, city)

with open("../data/clean/shelters_inventory_by_address.json", "w") as file:
    file.write(json.dumps(housing_inventory_json))
