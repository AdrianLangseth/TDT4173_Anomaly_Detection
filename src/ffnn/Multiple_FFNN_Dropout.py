from Multiple_FFNN import create_model


def create_dropout_model(size: int):
    create_model(size=size, is_drop_out=True)


if __name__ == '__main__':
    needed_model_sizes = [1000, 2500, 7000, 19000, 50000]
    for size in needed_model_sizes:
        create_dropout_model(size)

