import ipaddress

dataset_path = "data/public_dataset/sample_addresses.txt"
word_split_path = "data/processed_data/word_data.txt"
colon_removed_path = "data/processed_data/nocolon_data.txt"


def read_data():
    f = open(dataset_path, "r")
    raw_data = f.readlines()
    f.close()

    return raw_data


def word2vec_processing(nybbles):
    null = ""
    address = null.join(nybbles) + "\n"

    word2vec_data = []
    location_alpha = '0123456789abcdefghijklmnopqrstuv'
    for nybble, location in zip(address, location_alpha):
        word2vec_data.append(nybble + location)
    return word2vec_data


def data_processing(raw_data):
    word_split_list = []
    colon_removed_list = []

    space = " "
    null = ""
    for address in raw_data:
        address = ipaddress.ip_address(address[:-1]).exploded
        nybbles = address.split(":")
        word2vec_data = word2vec_processing(nybbles)
        word_split_list.append(space.join(word2vec_data) + "\n")
        # word_split_list.append(space.join(nybbles) + "\n")
        colon_removed_list.append(null.join(nybbles) + "\n")

    return word_split_list, colon_removed_list


def data_save(word_split_list, colon_removed_list):
    f = open(word_split_path, "w")
    f.writelines(word_split_list)
    f.close()

    f = open(colon_removed_path, "w")
    f.writelines(colon_removed_list)
    f.close()


if __name__ == "__main__":

    raw_data = read_data()
    word_split_list, colon_removed_list = data_processing(raw_data)
    data_save(word_split_list, colon_removed_list)
