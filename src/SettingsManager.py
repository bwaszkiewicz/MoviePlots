import json
import os.path


def load_settings():
    if not os.path.isfile('settings.txt'):
        settings_data = {"createCSV": 'True',
                         "outputFileCSV": 'standardized.csv',
                         "numberOfPlotsPerGenre": '400',
                         "numberOfInputWords": '200',
                         "vocabulary_size": '0',
                         "epochsNumber": '5',
                         "historyBatchSize": '25',
                         "resultsBatchSize": '50',
                         "learningFactor": '0.7'
                         }
        with open('settings.txt', 'w') as settings_file:
            settings_file.write(json.dumps(settings_data))
            settings_file.close()

    with open('settings.txt', 'r') as settings_file:
        json_settings = settings_file.read()
        settings_file.close()
        data = json.loads(json_settings)

    create_csv = str2bool(data["createCSV"])
    output_file_csv = data["outputFileCSV"]
    number_of_plots_per_genre = int(data["numberOfPlotsPerGenre"])
    number_of_input_words = int(data["numberOfInputWords"])
    vocabulary_size = int(data["vocabulary_size"])
    epochs_number = int(data["epochsNumber"])
    history_batch_size = int(data["historyBatchSize"])
    results_batch_size = int(data["resultsBatchSize"])
    learning_factor = float(data["learningFactor"])

    return create_csv, output_file_csv, number_of_plots_per_genre, number_of_input_words, vocabulary_size, epochs_number, history_batch_size, results_batch_size, learning_factor



def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def get_settings():
    with open('settings.txt', 'r') as settings_file:
        json_settings = settings_file.read()
        settings_file.close()
        data = json.loads(json_settings)
    return data


def set_settings(data):
    with open('settings.txt', 'w') as settings_file:
        settings_file.write(json.dumps(data))
        settings_file.close()


def change_setting(key, new_value):
    with open('settings.txt', 'r') as settings_file:
        json_settings = settings_file.read()
        settings_file.close()
        data = json.loads(json_settings)
        data[key] = new_value

    with open('settings.txt', 'w') as settings_file:
        settings_file.write(json.dumps(data))
        settings_file.close()