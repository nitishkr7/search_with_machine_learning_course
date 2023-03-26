import fasttext
import argparse
import csv
model = fasttext.load_model("/workspace/datasets/fasttext/title_model_epoch.bin")

parser = argparse.ArgumentParser(description='Process synonym generation')
general = parser.add_argument_group("general")
general.add_argument("--input", default="/workspace/datasets/fasttext/top_words.txt",  help="Input file containing words")
general.add_argument("--output", default="/workspace/datasets/fasttext/synonyms.csv", help="output file")
general.add_argument("--threshold", default=0.75,  help="Minimum threshold for similar words")

args = parser.parse_args()
input_file = args.input
output_file = args.output
threshold = args.threshold

f = open(input_file)
words = f.read().splitlines()

with open(output_file, 'w') as output:
    for word in words:
        nn_word_pair = model.get_nearest_neighbors(word)
        output.write(f'{word}')
        for (score, synonym_word) in nn_word_pair:
            if score >= threshold:
                output.write(f',{synonym_word}')
        output.write('\n')

