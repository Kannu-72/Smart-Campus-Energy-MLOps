
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from ml_workflow import (
    load_validate_data,
    train_evaluate_model,
    save_results
)


def load_data_task():
    data_path = load_validate_data()
    return str(data_path)


def train_model_task(**context):
    data_path = context["ti"].xcom_pull(
        task_ids="load_validate"
    )
    metrics = train_evaluate_model(data_path)
    return metrics


def save_results_task(**context):
    metrics = context["ti"].xcom_pull(
        task_ids="train_evaluate"
    )
    return save_results(metrics)


with DAG(
    dag_id="smart_campus_energy_workflow",
    description="Smart Campus Energy ML Workflow",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["MLOps", "Smart Campus"]
) as dag:

    task1 = PythonOperator(
        task_id="load_validate",
        python_callable=load_data_task
    )

    task2 = PythonOperator(
        task_id="train_evaluate",
        python_callable=train_model_task
    )

    task3 = PythonOperator(
        task_id="save_results",
        python_callable=save_results_task
    )

    task1 >> task2 >> task3
