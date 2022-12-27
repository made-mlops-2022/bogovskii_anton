import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


default_args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'retries': 0,
}


REPO_ROOT = '/home/bgvsk/Documents/made/2022/1_mlops/bogovskii_anton/hw3'


def docker_operator(dag, task_id, command):
    return DockerOperator(
        image='mlops-hw3-model',
        task_id=task_id,
        command=command,
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=os.path.join(REPO_ROOT, 'data'), target='/data', type='bind')],
        dag=dag,
    )


with DAG("model_train", default_args=default_args, schedule_interval='@weekly', start_date=days_ago(5), tags=['hw3']) as dag:
    preprocess = docker_operator(
        dag, 'preprocess-data',
        'preprocess --raw /data/raw/{{ ds }} --processed /data/processed/{{ ds }}',
    )

    split = docker_operator(
        dag, 'split-data',
        'split --data /data/processed/{{ ds }}',
    )

    train = docker_operator(
        dag, 'train-model',
        'train --data /data/processed/{{ ds }} --model /data/models/{{ ds }}',
    )

    validate = docker_operator(
        dag, 'validate-model',
        'validate --data /data/processed/{{ ds }} --model /data/models/{{ ds }}'
    )

    preprocess >> split >> train >> validate
