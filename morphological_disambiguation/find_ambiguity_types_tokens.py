import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

output_directory = "results"
os.makedirs(output_directory, exist_ok=True)


def plot_analysis(language, unique_tokens, ambiguity_types, ambiguity_tokens, types_percent, tokens_percent):
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    fig.suptitle(f"Analysis for '{language}' Language", fontsize=16)
    plt.subplots_adjust(hspace=.5)
    ax1.grid()
    ax2.grid()

    ax1.scatter(unique_tokens, ambiguity_types, color='r', label='analysis')
    ax2.scatter(unique_tokens, ambiguity_tokens, color='b', label='frequencies')

    ax2.set_xlabel('Unique tokens', fontsize=12)
    ax1.set_ylabel('Type analysis', fontsize=12)
    ax2.set_ylabel('Token Analysis', fontsize=12)
    ax1.title.set_text('Type Analysis')
    ax2.title.set_text('Token Analysis')
    plt.savefig(f"{output_directory}/{language}.png")


def ambuiguity_types_tokens(languages_directory):
    languages_directory = languages_directory

    output = []
    for subdir, dirs, files in os.walk(languages_directory):
        if subdir == languages_directory:
            continue
        types = {}
        frequencies = {}
        total_tokens = 0
        for file in files:
            for line in open(os.path.join(subdir, file)).readlines():
                line = line.strip()
                if line == '' or line[0] == '#':
                    continue
                total_tokens += 1
                row = line.split('\t')
                if '.' in row[0] or '-' in row[0]:
                    continue

                wordform = row[1]
                analysis = [row[2] + "/" + row[3] + "/" + row[5]]

                if wordform not in types:
                    types[str(wordform)] = set()
                types[str(wordform)].update(analysis)

                if wordform not in frequencies:
                    frequencies[str(wordform)] = 0
                frequencies[str(wordform)] += 1

        ambiguity_types = 0
        ambiguity_tokens = 0
        analysis_items = []
        frequency_items = []
        for word, word_analysis in types.items():
            analysis_items.append(len(word_analysis))
            frequency_items.append(frequencies[word])
            if len(word_analysis) > 1:
                ambiguity_types += 1
                ambiguity_tokens += frequencies[word]

        types = sorted(types.items(), key=lambda x: len(x[1]))
        with open(f"{output_directory}/{subdir.split('/')[-1]}_types.txt", "w") as outfile:
            outfile.write(str(types))
        frequencies = sorted(frequencies.items(), key=lambda x: x[1])
        with open(f"{output_directory}/{subdir.split('/')[-1]}_frequencies.txt", "w") as file:
            file.write(str(frequencies))
            # json.dump(frequencies, file)

        language = subdir.split('/')[-1]
        total_tokens = total_tokens
        unique_tokens = len(types)
        types_percentage = (ambiguity_types * 100) / unique_tokens
        tokens_percentage = (ambiguity_tokens * 100) / total_tokens
        # non_unique_tokens = total_tokens - len(types)
        output.append({"language": language,
                       "total_tokens": total_tokens,
                       "unique_tokens": unique_tokens,
                       # "non_unique_tokens": non_unique_tokens,
                       # "more_than_one_analysis": ambiguity_types,
                       "ambuiguity_types": types_percentage,
                       "ambuiguity_tokens": tokens_percentage})

        plot_analysis(language, np.array(range(len(types))), np.array(analysis_items), np.array(frequency_items),
                      types_percentage, tokens_percentage)
    return output


result = ambuiguity_types_tokens(languages_directory="languages")

print(result)
print(pd.DataFrame(result))
