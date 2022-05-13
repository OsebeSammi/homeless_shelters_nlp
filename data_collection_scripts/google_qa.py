import json
import googlemaps

api_key = "AIzaSyB3rZUfAkjl6d1B8wD5IDAVSP4B6RyjxGM"
# AIzaSyBwZu7uI5pfBbJ_U5GZY1YYmJsGBxSIYvk
maps = googlemaps.Client(key=api_key)

with open("../data/dataset.json", "r") as file:
    dataset = json.loads(file)

# for address, data in dataset.items():
#     # pull google map information for