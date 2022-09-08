import os

import pandas as pd

n_tokens = 0
n_analyses = 0
languages_directory = "languages"

output = []
for subdir, dirs, files in os.walk(languages_directory):
    types = {}
    frequencies = {}
    # analyses = {}
    total_tokens = 0
    for file in files:
        for line in open(os.path.join(subdir, file)).readlines():
            line = line.strip()
            if line == '':
                continue
            if line[0] == '#':
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
    for word, word_analysis in types.items():
        if len(word_analysis) > 1:
            ambiguity_types += 1
            ambiguity_tokens += frequencies[word]

    # with open(f"{subdir.split('/')[-1]}_types.txt", "w") as outfile:
    #     outfile.write(str(types))

    language = subdir.split('/')[-1]
    total_tokens = total_tokens
    unique_tokens = len(types)
    non_unique_tokens = total_tokens - len(types)
    if total_tokens:
        output.append({"language": language,
                       "total_tokens": total_tokens,
                       "unique_tokens": unique_tokens,
                       # "non_unique_tokens": non_unique_tokens,
                       # "more_than_one_analysis": ambiguity_types,
                       "ambuiguity_types": (ambiguity_types * 100) / unique_tokens,
                       "ambuiguity_tokens": (ambiguity_tokens * 100) / total_tokens})

print(output)
print(pd.DataFrame(output))
