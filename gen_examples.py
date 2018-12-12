import random

NUM_OF_EXAMPLES = 500
NUM_OF_DATA = 2000
POS_FILE = "pos_examples"
NEG_FILE = "neg_examples"
TRAIN_FILE = "data/train"
TEST_FILE = "data/test"


def get_char_sequence(character):
    """
    generate sequence of specific character with random size
    :return: string repr of the sequence
    """
    char_sequence = ""
    while True:
        char_sequence += character
        # 20% chance to stop concating similar chars.
        if random.randint(1, 5) == 1:
            break
    return char_sequence


def get_random_digits():
    """
    generate sequence of digits with random size and random choose of the digit
    :return: string repr of the sequence
    """
    digit_sequence = ""
    while True:
        digit = random.randint(1, 9)
        digit_sequence += str(digit)
        # 20% chance to stop concating digits.
        if random.randint(1, 5) == 1:
            break
    return digit_sequence


def generate_example(pos_flag):
    """
    generate 'positive' or 'negative' example depends on the pos flag
    :param pos_flag: boolean that if set as true so generate 'positive' example otherwise 'negative'
    :return: string repr of the example
    """
    # new generative sentence
    example = ""
    example += get_random_digits()
    example += get_char_sequence('a')
    example += get_random_digits()
    # pos flag set so 'b' comes before 'c'
    if pos_flag:
        example += get_char_sequence('b')
        example += get_random_digits()
        example += get_char_sequence('c')
    # otherwise negative example so 'c' comes before 'b'
    else:
        example += get_char_sequence('c')
        example += get_random_digits()
        example += get_char_sequence('b')
    example += get_random_digits()
    example += get_char_sequence('d')
    example += get_random_digits()
    return example


def generate_specific(n, pos_flag, file_path):
    """
    generating specific file of examples depends on pos flag
    :param n: number of examples to generate
    :param pos_flag: positive boolean flag, set true so generate positive otherwise examples
                    'negative' examples of the language
    :param file_path: file path + name of the file
    :return:
    """
    with open(file_path, 'w') as file:
        for i in range(n):
            new_example = generate_example(pos_flag)
            # stores fully generated example to it's file
            file.write(new_example + '\n')


def generate_randomly(n, file_path):
    """
    randomly generating examples of 'negative' and 'positive' examples of the language
    :param n: number of examples to generate
    :param file_path: file path + name of the file
    :return:
    """
    with open(file_path, 'w') as file:
        for i in range(n):
            # randomly choose 'true' or 'false'
            tag = random.choice([True, False])
            # generate example depends on the random choose
            new_example = generate_example(tag)
            file.write(new_example + '\t' + str(tag) + '\n')


def main():
    print("generating POS and NEG files...")
    # generate pos file
    generate_specific(NUM_OF_EXAMPLES, True, POS_FILE)
    # generate neg file
    generate_specific(NUM_OF_EXAMPLES, False, NEG_FILE)
    print("generating Train and Test files...")
    # generate test set (80% of data)
    generate_randomly(round(NUM_OF_DATA * 4/5), TRAIN_FILE)
    # generate test set (20% of data)
    generate_randomly(round(NUM_OF_DATA * 1/5), TEST_FILE)


if __name__ == "__main__":
    main()
