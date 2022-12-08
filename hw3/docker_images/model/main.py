import argparse
import preprocess
import split
import train
import validate


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    preprocess.init_subparser(subparsers.add_parser('preprocess'))
    split.init_subparser(subparsers.add_parser('split'))
    train.init_subparser(subparsers.add_parser('train'))
    validate.init_subparser(subparsers.add_parser('validate'))
    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
