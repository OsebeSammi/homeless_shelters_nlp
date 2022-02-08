import json

with open("data/clean/shelters.json", "r") as file:
    shelters = json.loads(file.read())

with open("data/clean/shelters_inventory_by_address.json", "r") as file:
    homeless_inventory = json.loads(file.read())

# by State
with open("data/clean/state_names.json", "r") as file:
    states = json.loads(file.read())

# search addresses
addresses = []
for key, inventory in homeless_inventory.items():
    addresses.append(key)

count = 0
for shelter in shelters:
    a = shelter["address"]
    flag = False
    for address in addresses:
        if str.lower(a) in str.lower(address) and len(a) > 0:
            count += 1
            flag = True
            break

    if not flag:
        print("Missed", a, "||", shelter["name"])

print("Found", count)

# shelter by state
# shelters_by_address = {}
# for shelter in shelters:
#     shelter["beds"] = ""
#     try:
#         address = formalize_address(shelter["address"], shelter["city"])
#
#         if address in shelters_by_address:
#             shelters_by_address[address]["description"].append(shelter["name"] + ". " + shelter["description"])
#         else:
#             shelter["description"] = [shelter["name"] + ". " + shelter["description"]]
#             shelters_by_address[address] = shelter
#     except Exception as error:
#         print("Missed", shelter["address"], shelter["city"])

    # if shelter["state"] in states:
    #     shelters_by_state[shelter["state"]].append(shelter)
