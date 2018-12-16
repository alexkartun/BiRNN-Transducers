import random

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}

NUM_OF_DATA = 2000
S_RANGE = 4
E_RANGE = 8
TRAIN_FILE = "m_power3_train"
TEST_FILE = "m_power3_dev"


def is_whole_cube(n):
    x = n ** (1 / 3)
    x = int(round(x))
    if x ** 3 == n:
        return True
    else:
        return False


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
    # n := number of repeating a's
    n = random.randint(S_RANGE, E_RANGE)
    whole_cube_n = n ** 3
    # new generative sentence
    example = ""
    example += get_char_sequence('a', whole_cube_n)
    return example


def generate_neg_example():
    """
    generate 'negative' example.
    :return: string of the example
    """
    # n := number of repeating a's
    n = random.randint(S_RANGE ** 3, E_RANGE ** 3)
    #  makes sure that randomized n is not a whole cube.
    while is_whole_cube(n):
        n = random.randint(S_RANGE ** 3, E_RANGE ** 3)
    example = ""
    example += get_char_sequence('a', n)
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
