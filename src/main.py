import mllib
import mllib.config as config

if __name__ == '__main__':
    print(f"{mllib.__version__=}")
    for k, v in config.items():
        print(f"{k}={v}")
