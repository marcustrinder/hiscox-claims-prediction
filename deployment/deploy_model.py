import sagemaker
from sagemaker.sklearn.estimator import SKLearn

def deploy_model(s3_input_data, model_output_path, role_arn):
    sagemaker_session = sagemaker.Session()
    sklearn = SKLearn(entry_point='deployment/train_model.py', role=role_arn, instance_type='ml.m5.large', framework_version='0.23-1')
    sklearn.fit({'train': s3_input_data})

    predictor = sklearn.deploy(instance_type='ml.m5.large', initial_instance_count=1)
    return predictor

if __name__ == "__main__":
    s3_input_data = 's3://your-bucket-name/path_to_your_training_data'
    model_output_path = 'models/xgboost_model.joblib'
    role_arn = 'arn:aws:iam::your-account-id:role/SageMakerExecutionRole'
    
    deploy_model(s3_input_data, model_output_path, role_arn)
