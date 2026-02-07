"""
Airflow DAG for daily model training.
Schedule: Daily at 2 AM
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['mlops@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_model_training',
    default_args=default_args,
    description='Daily training of ML models',
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False,
    tags=['ml', 'training']
)

# Task 1: Data quality check
check_data_quality = BashOperator(
    task_id='check_data_quality',
    bash_command='python mlops/monitoring/check_data_quality.py --train data/train.csv --test data/test.csv',
    dag=dag
)

# Task 2: Train all models
train_models = BashOperator(
    task_id='train_all_models',
    bash_command='python mlops/training/train_all_models.py',
    dag=dag
)

# Task 3: Evaluate models
evaluate_models = BashOperator(
    task_id='evaluate_models',
    bash_command='python mlops/monitoring/check_performance.py --model models/return_predictor_latest.pkl --test-data data/test.csv',
    dag=dag
)

# Task 4: Deploy if evaluation passes
deploy_models = BashOperator(
    task_id='deploy_models',
    bash_command='python mlops/deployment/deploy_model.py --model-path models/return_predictor_latest.pkl --env production',
    dag=dag
)

# Define task dependencies
check_data_quality >> train_models >> evaluate_models >> deploy_models
