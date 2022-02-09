import requests
import json

shelter_ids = range(50000, 60000)
shelter_data = {}
state_data = {}
malfunction = []
for shelter_id in shelter_ids:
    # try:
    data = requests.get("https://www.homelessshelterdirectory.org/get_listings_data.php?ids=" + str(shelter_id))

    if data.text == "":
        malfunction.append(shelter_id)
    else:
        if data.status_code == 200 and data.text != 'null':
            datum = data.json()[0]
            shelter_type = datum["type"]

            # create entry for shelter data
            if shelter_type not in shelter_data:
                shelter_data[shelter_type] = []

            # shelter data
            shelter = {
                "id": str(datum["listing_id"]),
                "name": datum["title"],
                "description": datum["description"].replace("\r", " ").replace("\n", " ").replace("<br />", " "),
                "address": datum["address"],
                "city": datum["city"],
                "county": datum["county"],
                "state": datum["state"],
                "zip": datum["zip"],
                "phone": datum["phone"],
                "web": datum["website"],
                "latitude": datum["latitude"],
                "longitude": datum["longitude"],
                "facebook": datum["facebook_url"],
                "type": datum["type"],
                "short_info": datum["blurb"]
            }
            shelter_data[shelter_type].append(shelter)

            # state stats
            state = datum["state"]

            if state not in state_data:
                state_data[state] = {}

            if shelter_type not in state_data[state]:
                state_data[state][shelter_type] = 0

            state_data[state][shelter_type] += 1

            print(shelter_id, datum["title"], "in ", state, datum["type"])
        # except Exception as error:
        #     print(error)

# write to file
with open("data_state.json", "w+") as file:
    file.write(json.dumps(state_data))

# write shelter
for shelter_type, shelters in shelter_data.items():
    with open("data_"+shelter_type+".json", "w+") as file:
        file.write(json.dumps(shelters))

with open("malfunction.txt", "w+") as file:
    file.write(str(malfunction))
