import random

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}

NUM_OF_DATA = 2000
S_RANGE = 15
E_RANGE = 50
TRAIN_FILE = "cross_train"
TEST_FILE = "cross_dev"


def get_char_sequence(character, n):
    """
    generate sequence of specific character with random size
    :return: string repr of the sequence
    """
    char_sequence = ""
    for _ in range(n):
        char_sequence += character
    return char_sequence


def generate_pos_example():
    """
    generate 'positive' example.
    :return: string of the example
    """
    # n := number of repeating a's and c's.
    # m := number of repeating b's and d's
    n = random.randint(S_RANGE, E_RANGE)
    m = random.randint(S_RANGE, E_RANGE)
    # new generative sentence
    example = ""
    example += get_char_sequence('a', n)
    example += get_char_sequence('b', m)
    example += get_char_sequence('c', n)
    example += get_char_sequence('d', m)
    return example


def generate_neg_example():
    """
    generate 'negative' example.
    :return: string of the example
    """
    # n := number of repeating a's and c's.
    # m := number of repeating b's and d's
    n = random.randint(S_RANGE, E_RANGE)
    m = random.randint(S_RANGE, E_RANGE)
    l = n
    k = m
    # validation that l, k are not randomized to be equal n, m. Even so, they would be pretty close in value.
    while l == n:
        l = random.randint(int(0.8 * n), int(1.2 * n))
    while k == m:
        k = random.randint(int(0.8 * m), int(1.2 * m))
    # new generative sentence
    example = ""
    example += get_char_sequence('a', n)
    example += get_char_sequence('b', m)
    example += get_char_sequence('c', l)
    example += get_char_sequence('d', k)
    return example


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
            if tag:
                new_example = generate_pos_example()
            else:
                new_example = generate_neg_example()
            file.write(new_example + '\t' + str(tag) + '\n')


def main():
    print("generating Train file...")
    # generate test set (80% of data)
    generate_randomly(round(NUM_OF_DATA * 4/5), TRAIN_FILE)
    print("generating Dev file...")
    # generate test set (20% of data)
    generate_randomly(round(NUM_OF_DATA * 1/5), TEST_FILE)


if __name__ == "__main__":
    main()
