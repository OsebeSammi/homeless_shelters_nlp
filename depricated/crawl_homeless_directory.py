import json
import requests
import re
import ast

# load shelter data
shelter_data = []
# state_data = {}

# state data
with open("state_shelter_data.json", "r") as file:
    data = file.read()
    state_data = ast.literal_eval(data)

# state regex set up
state_link = "https://www.homelessshelterdirectory.org/"
state_regex = r"https.*state/.[a-z_]+"

# http requests
response = requests.get(state_link)
state_results = re.findall(state_regex, response.text)

# loop through states
shelter_count = 0
seen_addresses = []
try:
    for i in range(len(state_results)):
        state_name = state_results[i].split("/")[4].strip()

        # check whether state data has already been added
        if state_name in state_data:
            continue

        state_shelter_count = 0
        print("\nState", state_name)
        link = state_results[i]

        # city regex
        city_regex = r"https.*city/.[a-z_-]+"
        response = requests.get(link)
        city_results = re.findall(city_regex, response.text.split("<!-- End Main content -->")[0])

        # loop through city
        for city_link in city_results:
            response = requests.get(city_link)
            content_regex = '<div class="item_content">\s*<h4>.*</h4>\s*.*\s*<p>.*\s*.*</p>\s*<a.*</a>\s*<div.*>\s*<span.*>\s*</div>\s*</div>'
            shelter_results = re.findall(content_regex, response.text)

            # shelter content
            shelter_info_regex = '<a class="btn btn_red" href="(.+?)"><i class="fa fa-arrow-circle-right"></i>&nbsp; See more details</a>'
            for shelter in shelter_results:
                link_info = re.search(shelter_info_regex, shelter).group(1).strip()

                # check whether link has been crawled already
                if link_info in seen_addresses:
                    continue
                else:
                    seen_addresses.append(link_info)

                try:
                    shelter_details = requests.get(link_info)

                    # extract shelter information
                    # shelter name
                    name_regex = '<h1 class="entry_title">(.+?)</h1>'
                    shelter_name = re.search(name_regex, shelter_details.text).group(1).strip()

                    # shelter contact
                    address_regex = '<p><strong>Address</strong><br>\s*(.+?)<br>\s*(.+?)<br>\s*</p>'
                    address = re.search(address_regex, shelter_details.text)
                    street = address.group(1).strip()
                    city = address.group(2).strip()

                    # phone
                    phone_regex = '<i class="fa fa-phone"></i>(.+?)</a>'
                    phone = re.search(phone_regex, shelter_details.text).group(1).strip()

                    # shelter text
                    details = shelter_details.text.split("<!-- Entry content -->")[1]
                    details = details.split("<!-- End Entry content -->")[0]
                    details = details.split("<p>")[3].split("</p>")[0]

                    # remove html tags
                    details = details.replace("<br />", "\n")
                    details = details.replace("<div>", "")
                    details = details.replace("</div>", "")
                    details = details.replace("&amp;", "")
                    details = details.replace("&nbsp;", "")

                    # structure the data
                    shelter_entry = {
                        "name": shelter_name,
                        "address": street + " " + city,
                        "state": state_name,
                        "phone": phone,
                        "details": details,
                        "detail_size": len(details),
                        "details_link": link_info
                    }

                    # append
                    shelter_data.append(shelter_entry)

                    # counters
                    shelter_count += 1
                    state_shelter_count += 1
                    print(shelter_count, "Shelter", shelter_name)

                except BaseException as error:
                    print("---------------------------------------")
                    print("Could not crawl")
                    print("Link", link_info)
                    print("Link Trace", city_link)
                    print("---------------------------------------")

        state_data[state_name] = state_shelter_count

    print("total shelters", len(shelter_data))

except BaseException as error:
    print(error)

# write shelter data
with open("shelter_data.json", "w+") as file:
    file.write(json.dumps(shelter_data))

# write state data
with open("state_shelter_data.json", "w+") as file:
    file.write(json.dumps(state_data))
