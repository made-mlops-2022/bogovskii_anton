import argparse
import pandas as pd
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)

    assert requests.get(f'http://{args.host}:{args.port}/health').status_code == 200
    for row in df.values:
        data = {'vals': [float(val) for val in row][:-1]}
        response = requests.post(f'http://{args.host}:{args.port}/predict', json=data)
        assert response.status_code == 200
        print(response.json()['result'])


if __name__ == '__main__':
    main()
