from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

import os


default_args = {
    'owner':   'airflow',
    'email':   ['airflow@example.com'],
    'retries': 0,
}


REPO_ROOT = '/home/bgvsk/Documents/made/2022/1_mlops/bogovskii_anton'


with DAG(
    "data_synthesis",
    default_args=default_args,
    schedule_interval='@daily',
    start_date=days_ago(5),
    tags=['hw3'],
) as dag:
    generate_data = DockerOperator(
        image='mlops-hw3-data-synthesis',
        command='/data/raw/{{ ds }}/',
        task_id='generate-data',
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=os.path.join(REPO_ROOT, 'hw3/data'), target='/data', type='bind')],
        dag=dag,
    )
