
def get_words(descriptions):
    words = []
    for description in descriptions:
        for word in description["description"].split(" "):
            if len(word) > 0:
                words.append(word)

    return words
