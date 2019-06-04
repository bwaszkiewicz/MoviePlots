def cut_movies(data, count):
    print("Shuffle init")
    genres_count_dictionary = count_genres(data)

    shuffled_data = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])

    for genre in genres_count_dictionary:
        if genres_count_dictionary[genre] < count:
            print("Can't shuffle on this count")
            return
    for genre in genres_count_dictionary:
        for i in range(count):
            shuffled_data = processed_data.append(data.loc[i, :], ignore_index=True) # TODO przerobic

        # TODO

    return



def count_genres(data):
    genres_count_dictionary = {
        "thriller": 0,
        "science_fiction": 0,
        "romance": 0,
        "musical": 0,
        "horror": 0,
        "drama": 0,
        "crime": 0,
        "comedy": 0,
        "animation": 0,
        "adventure": 0,
        "action": 0}
    for i in range(0, len(data)):
        corrected_genre = data.loc[i, :].GenreCorrected
        if corrected_genre == "thriller":
            genres_count_dictionary["thriller"] += 1

        elif corrected_genre == "science_fiction":
            genres_count_dictionary["science_fiction"] += 1

        elif corrected_genre == "romance":
            genres_count_dictionary["romance"] += 1

        elif corrected_genre == "musical":
            genres_count_dictionary["musical"] += 1

        elif corrected_genre == "horror":
            genres_count_dictionary["horror"] += 1

        elif corrected_genre == "drama":
            genres_count_dictionary["drama"] += 1

        elif corrected_genre == "crime":
            genres_count_dictionary["crime"] += 1

        elif corrected_genre == "comedy":
            genres_count_dictionary["comedy"] += 1

        elif corrected_genre == "animation":
            genres_count_dictionary["animation"] += 1

        elif corrected_genre == "adventure":
            genres_count_dictionary["adventure"] += 1

        elif corrected_genre == "action":
            genres_count_dictionary["action"] += 1

    return genres_count_dictionary

def seperate_genres(data):

    # TODO

    return