import re
import requests
import json
import uszipcode as uszip

# us zip search
search_zip = uszip.SearchEngine()

# load root website
listings_url = "https://www.shelterlistings.org"
listings_response = requests.get(listings_url)

# regex extract state data links
state_regex = r"https[:][/][/]www[.]shelterlistings[.]org[/]state[/].*[.]html"
state_links = re.findall(state_regex, listings_response.text)

# vars
shelter_data = []
seen_links = []
manual_convert = []

with open("data_state_2.json", "r") as file:
    state_data = json.loads(file.read())

try:
    for state_link in state_links:
        state = state_link.split("/")[4].replace(".html", "")

        if state in state_data:
            print("Done State", state)
            continue

        state_response = requests.get(state_link)

        # extract city data links
        city_regex = r"https[:][/][/]www[.]shelterlistings[.]org[/]city[/][a-z-_]+[.]html"
        city_links = re.findall(city_regex, state_response.text)

        for city_links in city_links:
            city_response = requests.get(city_links)

            # extract shelter links
            shelter_regex = r"https[:][/][/]www[.]shelterlistings[.]org[/]details[/][1-9]+[/]"
            shelter_links = re.findall(shelter_regex, city_response.text)

            for shelter_link in shelter_links:
                if shelter_link in seen_links:
                    continue
                else:
                    seen_links.append(shelter_link)

                shelter_response = requests.get(shelter_link)
                text = shelter_response.text

                # extract shelter data
                details = text.split("<!--#main-menu-->")
                details = details[1].split("</script>")
                data = details[0].replace('<div class="wrapper">', '') \
                    .replace('<div id="content" class="grid8 first">', '') \
                    .replace('<script type="application/ld+json">', '')
                data = data.replace("\n", "").replace("\t", "")
                try:
                    data = json.loads(data, strict=False)
                except Exception as error:
                    manual_convert.append(data)
                    continue

                # confirm address not already seen
                address = data["address"]["streetAddress"]

                # shelter data
                # phone
                phone_regex = r"[0-9][0-9][0-9].*[0-9][0-9][0-9][-.]+[0-9][0-9][0-9][0-9]"

                try:
                    phone = re.search(phone_regex, text).group()
                except Exception as error:
                    phone = ""

                # id
                shelter_id = shelter_link.split("/")[4]

                # fetch zip code data
                post_code = data["address"]["postalCode"]
                location_info = search_zip.by_zipcode(post_code)

                if location_info is None:
                    county = lat = lng = ""
                    print("----------")
                    print("No Location Data", data["name"])
                    print(shelter_link)
                    print("----------")
                else:
                    county = location_info.county
                    lat = location_info.lat
                    lng = location_info.lng

                shelter = {
                    "id": shelter_id,
                    "name": data["name"],
                    "description": data["description"],
                    "address": data["address"]["streetAddress"],
                    "city": data["address"]["addressLocality"],
                    "county": county,
                    "state": state,
                    "zip": data["address"]["postalCode"],
                    "phone": phone,
                    "web": shelter_link,
                    "latitude": lat,
                    "longitude": lng,
                    "facebook": "",
                    "type": "shelter",
                    "short_info": ""
                }

                # populate state data
                if state not in state_data:
                    state_data[state] = 0

                state_data[state] += 1

                # populate shelter data
                shelter_data.append(shelter)

                print("Shelter", data["name"], shelter_link)
except Exception as error:
    print(error)

# write to file
with open("data_state_2.json", "w+") as file:
    file.write(json.dumps(state_data))

# write shelter
with open("data_shelter_2.json", "w+") as file:
    file.write(json.dumps(shelter_data))

with open("manual.txt", "w+") as file:
    for data in manual_convert:
        file.write(data)
        file.write("\n")

