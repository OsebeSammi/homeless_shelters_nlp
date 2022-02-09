import json

with open("../data/clean/shelters_by_address.json", "r") as file:
    shelters_1 = json.loads(file.read())

with open("../data/clean/shelters_2_by_address.json", "r") as file:
    shelters_2 = json.loads(file.read())

with open("../data/clean/shelters_inventory_by_address.json", "r") as file:
    inventory = json.loads(file.read())

with open("../data/clean/hp_by_address.json", "r") as file:
    h_programs = json.loads(file.read())

final_dataset = {}

# check shelters 1
counter = 0
print("Counter")
inventory_keys = inventory.keys()
for shelter_key, shelter in shelters_1.items():
    if shelter_key in inventory_keys:
        shelter["beds"] = str(inventory[shelter_key]["beds"])
        shelter["verified"] = True
        counter += 1
    else:
        shelter["beds"] = ""
        shelter["verified"] = False
    final_dataset[shelter_key] = shelter

print("Verified", counter)

# check shelters 2
for shelter_key, shelter in shelters_2.items():
    if shelter_key in inventory:
        shelter["beds"] = str(inventory[shelter_key]["beds"])
        shelter["verified"] = True
    else:
        shelter["beds"] = ""
        shelter["verified"] = False

    if shelter_key not in final_dataset:
        final_dataset[shelter_key] = shelter
        if shelter["verified"]:
            counter += 1

print("Verified", counter)
with open("../data/dataset.json", "w") as file:
    file.write(json.dumps(final_dataset))

# housing programs
# hp = {}
# for hp_key, housing_program in h_programs.items():
#     if hp_key in inventory:
#         housing_program["beds"] = str(inventory[hp_key]["beds"])
#         housing_program["verified"] = True
#         counter += 1
#     else:
#         housing_program["beds"] = ""
#         housing_program["verified"] = False
#     housing_program["type"] = "housing program"
#     hp[hp_key] = housing_program
#
# print("Verified", counter)
# with open("data/clean/hp_by_address.json", "w") as file:
#     file.write(json.dumps(hp))
