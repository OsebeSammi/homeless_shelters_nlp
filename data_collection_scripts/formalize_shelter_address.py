import json
import googlemaps

api_key = ""  # TODO refactor to take this from config
maps = googlemaps.Client(key=api_key)


# Formalize Address
def formalize_address(address, city):
    text = address + " " + city
    address = maps.places_autocomplete(input_text=text, types="address", components={"country": ["US"]})[0]
    return address["description"]


with open("../data/clean/housing_programs.json", "r") as file:
    shelters = json.loads(file.read())

count = 0
shelters_by_address = {}
for shelter in shelters:
    address = shelter["address"]
    city = shelter["city"]
    description_temp = shelter["description"]
    shelter["description"] = [{
        "title": shelter["name"],
        "description": shelter["description"]
    }]
    temp = str.lower(address.replace(".", ""))
    if "po box" not in temp:
        try:
            formalized_address = formalize_address(address, city)
            count += 1
            print("API REQ", count)

            if formalized_address in shelters_by_address:
                shelters_by_address[formalized_address]["description"].append({
                    "title": shelter["name"],
                    "description": description_temp
                })
            else:
                shelters_by_address[formalized_address] = shelter
        except Exception as e:
            print("Missed ", address, city)

with open("../data/clean/hp_by_address.json", "w") as file:
    file.write(json.dumps(shelters_by_address))

