# Hiscox Claims Prediction
#
## Project Overview

This project aims to determine the likelihood of claims for insurance applications using machine learning. The project involves extracting functionalities from a Jupyter notebook, setting up a CI/CD pipeline, and deploying the model to a cloud service.

## Project Structure

hiscox-claims-prediction/
├── .github/
│ └── workflows/
│ └── ci-cd.yaml
├── data/
├── models/
├── notebooks/
├── src/
│ ├── init.py
│ ├── data_processing.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ ├── predict.py
├── tests/
├── scripts/
│ ├── train_model.py
│ ├── evaluate_model.py
│ ├── predict.py
├── deployment/
| ├── sg_register_model.py
| ├── sg_create_group.py
│ ├── train_model.py
│ ├── deploy_model.py
├── requirements.txt
├── setup.py
├── README.md


## Setup Instructions

### Prerequisites

- Python 3.8+
- AWS Account (for SageMaker deployment)
- GitHub account (for CI/CD)
- AWS CLI


1. **Install AWS CLI**:

   If you haven't installed the AWS CLI, follow the [installation guide](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).

2. **Configure AWS CLI**:

   Run the following command and follow the prompts to configure your AWS CLI:

   ```sh
   aws configure

    Example configuration:

    AWS Access Key ID [None]: YOUR_ACCESS_KEY_ID
    AWS Secret Access Key [None]: YOUR_SECRET_ACCESS_KEY
    Default region name [None]: us-east-1
    Default output format [None]: json

3. **Create IAM Role for SageMaker**:

Go to the IAM Console, and create a new role with the following steps:

Select trusted entity: Choose "AWS service".
Choose the service that will use this role: Select "SageMaker".
Attach permissions policies: Attach the following policies:
- AmazonSageMakerFullAccess
- AmazonS3FullAccess
Role name: Give your role a name, e.g., SageMakerExecutionRole.
After creating the role, note down the Role ARN (Amazon Resource Name) which will be used in the deployment scripts.

4. **Upload Training Data to S3**:

Upload your training data to an S3 bucket. For example:

aws s3 cp data/training_data s3://training_data/training_data_1

5. **Installation:**
Clone the repository:

    git clone https://github.com/your-username/hiscox-claims-prediction.git
    cd hiscox-claims-prediction

Install the required packages:

    pip install -r requirements.txt

Install the package:

    python setup.py install


6. **Create a GitHub repository and push the code**:

git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/username/hiscox-claims-prediction.git
git push -u origin main


7. **Deployment**:
AWS SageMaker Deployment
Training and Deployment Script:
Use the deployment/train_model.py script for training on SageMaker.

Deploy Model Using SageMaker:
Use the deployment/deploy_model.py script for deploying the trained model on SageMaker.

Example Deployment Script Usage:

Managing Model Versions
SageMaker Model Registry
Create a Model Package Group:

Use sg_create_group.py

Register a Model Version:

Use sg_register_model.py

Update Model Version:
For a new version (e.g., v2), repeat the registration process with updated model data and entry point.

8. **Versioning and secrets**:

Code versioning can be managed entirely in GitHub and AWS secrets can also be saved here or in AWS Secrets Manager.