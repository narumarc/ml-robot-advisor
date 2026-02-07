"""
Airflow DAG for drift monitoring.
Schedule: Every 6 hours
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator

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
    'drift_monitoring',
    default_args=default_args,
    description='Monitor data drift every 6 hours',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    catchup=False,
    tags=['ml', 'monitoring', 'drift']
)

# Task 1: Check data drift
check_drift = BashOperator(
    task_id='check_drift',
    bash_command='python mlops/monitoring/check_drift.py --reference data/reference.csv --current data/production.csv --output reports/drift_report.html',
    dag=dag
)

# Task 2: Generate drift report
generate_drift_report = BashOperator(
    task_id='generate_drift_report',
    bash_command='python mlops/monitoring/generate_reports.py --data reports/drift_data.json --output reports/drift_report.html',
    dag=dag
)

# Task 3: Alert if drift detected (conditional)
def check_drift_result(**context):
    """Check if drift was detected and branch accordingly."""
    # Read drift result from XCom or file
    # For simplicity, always go to alert
    return 'send_drift_alert'

branch_on_drift = BranchPythonOperator(
    task_id='branch_on_drift',
    python_callable=check_drift_result,
    provide_context=True,
    dag=dag
)

send_drift_alert = BashOperator(
    task_id='send_drift_alert',
    bash_command='echo "ALERT: Data drift detected!" | mail -s "Drift Alert" mlops@company.com',
    dag=dag
)

no_drift = BashOperator(
    task_id='no_drift',
    bash_command='echo "No drift detected"',
    dag=dag
)

# Define task dependencies
check_drift >> generate_drift_report >> branch_on_drift
branch_on_drift >> [send_drift_alert, no_drift]
