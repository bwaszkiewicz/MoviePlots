import pandas as pd
import random

def cut_movies(data, count):
    print("Shuffle init")
    genres_array = ["thriller", "science_fiction", "romance", "musical", "horror", "drama", "crime", "comedy", "animation", "adventure", "action"]
    shuffled_data = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    genres_seperate_dictionary = seperate_genres(data)

    for genre in genres_array:
        if len(genres_seperate_dictionary[genre].GenreCorrected) < count:
            print("Can't shuffle on this count")
            return None

    for genre in genres_array:
        draw_lots_array = draw_numbers(len(genres_seperate_dictionary[genre].GenreCorrected),count)
        for i in range(count):
                shuffled_data = shuffled_data.append(genres_seperate_dictionary[genre].loc[draw_lots_array[i], :], ignore_index=True) # TODO przerobic

    return shuffled_data

def draw_numbers(max_draw, count):
     return random.sample(range(0, max_draw), count)

def seperate_genres(data):

    only_thriller = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    only_science_fiction = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    only_romance = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    only_musical = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    only_horror = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    only_drama = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    only_crime = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    only_comedy = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    only_animation = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    only_adventure = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])
    only_action = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])

    for i in range(0, len(data)):
        corrected_genre = data.loc[i, :].GenreCorrected
        if corrected_genre == "thriller":
            only_thriller = only_thriller.append(data.loc[i, :], ignore_index=True)

        elif corrected_genre == "science_fiction":
            only_science_fiction = only_science_fiction.append(data.loc[i, :], ignore_index=True)

        elif corrected_genre == "romance":
            only_romance = only_romance.append(data.loc[i, :], ignore_index=True)

        elif corrected_genre == "musical":
            only_musical = only_musical.append(data.loc[i, :], ignore_index=True)

        elif corrected_genre == "horror":
            only_horror = only_horror.append(data.loc[i, :], ignore_index=True)

        elif corrected_genre == "drama":
            only_drama = only_drama.append(data.loc[i, :], ignore_index=True)

        elif corrected_genre == "crime":
            only_crime = only_crime.append(data.loc[i, :], ignore_index=True)

        elif corrected_genre == "comedy":
            only_comedy = only_comedy.append(data.loc[i, :], ignore_index=True)

        elif corrected_genre == "animation":
            only_animation = only_animation.append(data.loc[i, :], ignore_index=True)

        elif corrected_genre == "adventure":
            only_adventure = only_adventure.append(data.loc[i, :], ignore_index=True)

        elif corrected_genre == "action":
            only_action = only_action.append(data.loc[i, :], ignore_index=True)

    genres_seperate_dictionary = {
        "thriller": only_thriller,
        "science_fiction": only_science_fiction,
        "romance": only_romance,
        "musical": only_musical,
        "horror": only_horror,
        "drama": only_drama,
        "crime": only_crime,
        "comedy": only_comedy,
        "animation": only_animation,
        "adventure": only_adventure,
        "action": only_action
    }

    return genres_seperate_dictionary