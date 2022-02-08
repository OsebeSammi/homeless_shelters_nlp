import re
import requests
import json
import pandas as pd
import numpy as np


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
    description = description.replace("\n", "")
    description = description.replace("\r", "")

    return description


# load unique zips to lookup
housing_inventory = pd.read_excel("data/raw/2020-HIC-Raw-File.xlsx", sheet_name="HIC_RawData2020",
                                  usecols=["zip"], na_filter=False)
housing_inventory = housing_inventory.drop_duplicates(subset=["zip"])
zip_codes = housing_inventory.to_numpy()
print("Unique Zip Codes", len(zip_codes))

# crawl the data
url = "https://www.shelterlist.com/zip/"

# Search by Zip Home
shelters = {}
counter = 0

seen_links = []
for zip_code in zip_codes:
    print("ZIP", zip_code[0])
    try:
        shelters_response = requests.get(url+str(zip_code[0]))
        shelters_regex = r'"https[:][/][/]www[.]shelterlist[.]com[/]details[/][a-z0-9-_]+">Read Full Details'
        shelter_links = re.findall(shelters_regex, shelters_response.text)

        # Extract details from shelter page
        if len(shelter_links) == 0:
            print("NO SHELTERS", "\n")

        for shelter_link in shelter_links:
            if shelter_link in seen_links:
                continue

            seen_links.append(shelter_link)

            shelter_link = shelter_link.replace("\"", "")
            shelter_link = shelter_link.replace('>Read Full Details', "")
            counter += 1
            print(counter, shelter_link)
            shelter_response = requests.get(shelter_link)

            # extract name
            shelter_name_regex = r"<h2>(.*)</h2>\s*<script"
            try:
                shelter_name = re.search(shelter_name_regex, shelter_response.text).groups()[0]
                print("Name", shelter_name)
            except Exception:
                continue

            # extract address
            address_regex = r'<i class="fa fa-map-marker fa-2x"></i>\s*</div>\s*<div>(.*)</div>'
            try:
                address_info = re.search(address_regex, shelter_response.text).groups()[0].split("<br/>")
                address = address_info[0].strip()
                city = address_info[1].split(",")[0].strip()
                state_zip = address_info[1].split(",")[1].strip().split(" ")
                state = state_zip[0].strip()
                zip_ = state_zip[1].strip()
            except Exception:
                continue

            # extract phone number
            phone_regex = r'<i class="fa fa-phone fa-2x"></i>\s*</div>\s*<div>(.*)</div>'
            try:
                phone = re.search(phone_regex, shelter_response.text).groups()[0]
            except Exception:
                phone = ""


            # extract shelter details
            about_start = '<h3>About this Shelter</h3>'
            about_stop = '<ul class="list list-unstyled">'
            details = shelter_response.text.split(about_start)[1].split(about_stop)[0]
            description = remove_html(details).strip()

            # shelter data
            shelter = {
                "name": shelter_name,
                "description": description,
                "address": address,
                "city": city,
                "county": "",
                "state": state,
                "zip": zip_,
                "phone": phone,
                "web": "",
                "latitude": "",
                "longitude": "",
                "facebook": "",
                "type": "shelter",
                "short_info": ""
            }

            key = address + " " + city + " " + state
            if key not in shelters:
                shelters[key] = shelter
            else:
                print("present", key)

    except Exception as error:
        print(error)

with open("data/raw/shelterlist.json", "w") as file:
    file.write(json.dumps(shelters))
