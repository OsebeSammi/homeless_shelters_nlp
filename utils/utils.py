
def get_words(descriptions):
    words = []
    for word in descriptions.split(" "):
        if len(word) > 0:
            words.append(word)

    return words
