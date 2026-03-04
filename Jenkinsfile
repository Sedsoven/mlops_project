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
                bat '"C:\\Users\\tussi\\AppData\\Local\\Python\\bin\\python.exe" -m venv venv'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'venv\\Scripts\\python -m pip install --upgrade pip'
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
                bat 'venv\\Scripts\\python train.py'
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: '*.pkl, *.json', fingerprint: true
            }
        }

    }
}