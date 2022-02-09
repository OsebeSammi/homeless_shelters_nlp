import json
import re

# state names
with open("../data/clean/state_names.json", "r") as file:
    state_names = json.loads(file.read())
state_short = {}
for key, val in state_names.items():
    state_short[str.lower(val)] = key

# load data crawled from https://www.homelessshelterdirectory.org/
with open("../data/raw/data_shelter.json", "r") as file:
    homeless_shelter_directory_data = json.loads(file.read())

# load data crawled from https://www.shelterlistings.org/
with open("../data/raw/data_shelter_3.json", "r") as file:
    shelter_listings_1 = json.loads(file.read())

with open("../data/raw/data_shelter_2.json", "r") as file:
    shelter_listings_2 = json.loads(file.read())

# join shelter listing data
for i in range(len(shelter_listings_2)):
    flag = False
    for j in range(5000, 6013):
        if shelter_listings_2[i]["name"] == shelter_listings_1[j]["name"] and \
                shelter_listings_2[i]["zip"] == shelter_listings_1[j]["zip"]:
            flag = True
            break
    if not flag:
        shelter_listings_1.append(shelter_listings_2[i])

# Remove Housing Authorities
shelter_listings = []
housing_authorities = []
housing_authority = "housing authority"
hud_text = "This is a HUD Housing approved Housing Assistance agency"
for shelter in shelter_listings_1:
    if housing_authority in str.lower(shelter["name"]) or hud_text in shelter["description"]:
        print("Excluding", shelter["name"], shelter["description"][:70], "...\n")
        housing_authorities.append(shelter)
    else:
        shelter_listings.append(shelter)


# remove html tags
def remove_html(description):
    # open tag
    regex_open = r"<.+?>"
    tags = re.findall(regex_open, description)
    for tag in tags:
        description = description.replace(tag, " ")
    description = description.replace("&nbsp;", "")
    description = description.replace("&amp;", "")
    description = description.replace("<br />", "")

    return description


# join the 2 data sources
for i in range(len(homeless_shelter_directory_data)):
    a = homeless_shelter_directory_data[i]
    a["description"] = remove_html(a["description"])

    flag = False
    for j in range(len(shelter_listings)):
        b = shelter_listings[j]
        b["description"] = remove_html(b["description"])

        # clean shelter data
        if ")" in b["phone"] and "(" not in b["phone"]:
            b["phone"] = "(" + b["phone"]

        if b["county"] is not None:
            b["county"] = b["county"].replace(" County", "")

        index = str.lower(b["state"])
        if len(index) == 2:
            b["state"] = str.upper(index)
        else:
            b["state"] = state_short[index]

        if (str.upper(a["name"]) == str.upper(b["name"]) or str.upper(a["address"]) == str.upper(b["address"])) and \
                (str.upper(a["state"]) == str.upper(b["state"]) or str.upper(a["county"]) == str.upper(b["county"])):
            flag = True
            # update description
            b["description"] = b["description"] + " " + a["description"]
            break

        # update
        shelter_listings[j] = b

    if not flag:
        shelter_listings.append(a)

with open("../data/clean/shelters.json", "w") as file:
    file.write(json.dumps(shelter_listings))

with open("../data/clean/housing_programs.json", "w") as file:
    file.write(json.dumps(housing_authorities))