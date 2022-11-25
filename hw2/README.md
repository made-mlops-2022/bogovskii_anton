# Homework 2

To build & run docker image locally, do
```bash
cd ${REPO_ROOT}/hw2
docker build -t mlops-hw2 .
docker run --net=host mlops-hw2
```

To pull & run docker image, do
```bash
cd ${REPO_ROOT}/hw2
docker pull abogovski/mlops-hw2
docker run --net=host mlops-hw2
```

To run unittest do
```
pip -r requirements_all.txt
pytest .
```
