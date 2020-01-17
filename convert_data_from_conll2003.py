import argparse, os
from file_operations import read_lines, write_lines
from build_vocab import make_vocabs
from build_glove import make_embeddings

TEST_A_CONLL2003_FILE_NAME = 'test.txt'
TEST_B_CONLL2003_FILE_NAME = 'dev.txt'
TRAIN_CONLL2003_FILE_NAME = 'train.txt'

def reformat_file(filename):
    lines = read_lines(filename)
    sentence = []
    words = []
    labels = []
    for line in lines[2:]:
        if line == '':
            words.append(' '.join(map(lambda token: token.split(' ')[0], sentence)))
            labels.append(' '.join(map(lambda token: token.split(' ')[-1], sentence)))
            sentence = []
        else:
            sentence.append(line)
    return words, labels


def convert(input_folder, output_folder):
    # Change format
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    words, tags = reformat_file(f'{input_folder}/{TEST_A_CONLL2003_FILE_NAME}')
    write_lines(f'{output_folder}/testa.words.txt', words)
    write_lines(f'{output_folder}/testa.tags.txt', tags)

    words, tags = reformat_file(f'{input_folder}/{TEST_B_CONLL2003_FILE_NAME}')
    write_lines(f'{output_folder}/testb.words.txt', words)
    write_lines(f'{output_folder}/testb.tags.txt', tags)

    words, tags = reformat_file(f'{input_folder}/{TRAIN_CONLL2003_FILE_NAME}')
    write_lines(f'{output_folder}/train.words.txt', words)
    write_lines(f'{output_folder}/train.tags.txt', tags)

    # Build vocabs
    make_vocabs(output_folder)

    # Build embeddings
    make_embeddings(output_folder, '/home/dima/models/ArModel100w2v.txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', type=str, default='/home/dima/tener/data/conll2003ru-bio-super-distinct')
    parser.add_argument('--output_folder', type=str, default='data/conll2003ru')

    args = parser.parse_args()
    convert(args.input_folder, args.output_folder)
