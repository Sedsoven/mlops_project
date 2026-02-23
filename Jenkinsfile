pipeline {
    agent any

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/Sedsoven/mlops_project.git'
            }
        }

        stage('Create Virtual Environment') {
            steps {
                bat 'python -m venv venv'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'venv\\Scripts\\pip install -r requirements.txt'
            }
        }

        stage('Data Check') {
            steps {
                bat 'venv\\Scripts\\python exp1_data_check.py'
            }
        }

        stage('Train Model') {
            steps {
                bat 'venv\\Scripts\\python vertex_train.py'
            }
        }

        stage('Compare Metrics') {
            steps {
                bat 'venv\\Scripts\\python compare_metrics.py'
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: '*.pkl', fingerprint: true
                archiveArtifacts artifacts: 'metrics.json'
                bat 'copy metrics.json previous_metrics.json'
            }
        }
    }
}
