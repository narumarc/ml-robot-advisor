"""
Airflow DAG for automated model retraining.
Triggered when drift or performance degradation detected.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['mlops@company.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'automated_retraining',
    default_args=default_args,
    description='Automated model retraining pipeline',
    schedule_interval=None,  # Triggered manually or by other DAGs
    catchup=False,
    tags=['ml', 'retraining', 'automation']
)

# Task 1: Wait for drift detection to complete
wait_for_drift_check = ExternalTaskSensor(
    task_id='wait_for_drift_check',
    external_dag_id='drift_monitoring',
    external_task_id='check_drift',
    timeout=600,
    poke_interval=60,
    dag=dag
)

# Task 2: Evaluate retraining triggers
evaluate_triggers = BashOperator(
    task_id='evaluate_retraining_triggers',
    bash_command='python mlops/retraining/retrain_trigger.py',
    dag=dag
)

# Task 3: Branch based on trigger decision
def check_retrain_decision(**context):
    """Check if retraining is needed."""
    # In practice, read from XCom or file
    # For demo, assume retraining needed
    return 'start_retraining'

branch_retrain = BranchPythonOperator(
    task_id='branch_retrain_decision',
    python_callable=check_retrain_decision,
    provide_context=True,
    dag=dag
)

# Task 4: Execute retraining pipeline
start_retraining = BashOperator(
    task_id='start_retraining',
    bash_command='python mlops/retraining/auto_retrain_pipeline.py',
    dag=dag
)

# Task 5: Skip retraining
skip_retraining = BashOperator(
    task_id='skip_retraining',
    bash_command='echo "Retraining not needed"',
    dag=dag
)

# Task 6: Validate new model
validate_model = BashOperator(
    task_id='validate_new_model',
    bash_command='python mlops/monitoring/check_performance.py --model models/return_predictor_latest.pkl --test-data data/test.csv',
    dag=dag
)

# Task 7: Deploy new model
deploy_new_model = BashOperator(
    task_id='deploy_new_model',
    bash_command='python mlops/deployment/deploy_model.py --model-path models/return_predictor_latest.pkl --env production',
    dag=dag
)

# Task 8: Update model registry
update_registry = BashOperator(
    task_id='update_model_registry',
    bash_command='python mlops/deployment/model_versioning.py',
    dag=dag
)

# Task 9: Send completion notification
send_notification = BashOperator(
    task_id='send_completion_notification',
    bash_command='echo "Retraining pipeline completed" | mail -s "Retraining Complete" mlops@company.com',
    dag=dag
)

# Define task dependencies
wait_for_drift_check >> evaluate_triggers >> branch_retrain
branch_retrain >> [start_retraining, skip_retraining]
start_retraining >> validate_model >> deploy_new_model >> update_registry >> send_notification
