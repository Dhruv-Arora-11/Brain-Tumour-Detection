pipeline {
    agent any

    triggers {
        githubPush()
    }

    stages {

        stage('Pull Latest Image') {
            steps {
                bat 'docker pull dhruv1472005/brain_tumor_detection:latest'
            }
        }

        stage('Run Container') {
            steps {
                bat '''
                docker stop brain_app 2>nul
                docker rm brain_app 2>nul
                docker run -d -p 8000:8000 --name brain_app dhruv1472005/brain_tumor_detection:latest
                '''
            }
        }
    }
}