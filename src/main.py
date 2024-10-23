import mllib

if __name__ == '__main__':
    print(f"{mllib.__version__=}")
    for k, v in mllib.config.items():
        print(f"{k}={v}")
