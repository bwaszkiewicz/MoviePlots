import time
import pandas as pd

def genres_filtr(data):
    processed_data = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])

    start_time = time.time()
    for i in range(0, len(data)):
        correctedgenre = data.loc[i, :].GenreCorrected
        if correctedgenre == "thriller" \
                or correctedgenre == "science_fiction" \
                or correctedgenre == "romance" \
                or correctedgenre == "musical" \
                or correctedgenre == "horror" \
                or correctedgenre == "drama" \
                or correctedgenre == "crime" \
                or correctedgenre == "comedy" \
                or correctedgenre == "animation" \
                or correctedgenre == "adventure" \
                or correctedgenre == "action":
            processed_data = processed_data.append(data.loc[i, :], ignore_index=True)
        print("Processing: " + str(i) + "/" + str(len(data)))
    elapsed_time = time.time() - start_time
    print("#GenresFiltr Execution time: " + str(elapsed_time))

    # processed_data.to_csv('processed_data.csv', ',')
    return processed_data
