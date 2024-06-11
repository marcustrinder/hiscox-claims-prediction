from sagemaker.model import Model

model = Model(
    model_data='s3://your-bucket-name/path_to_your_model.tar.gz',
    role='arn:aws:iam::your-account-id:role/SageMakerExecutionRole',
    entry_point='deployment/train_model.py'
)

model_package = model.register(
    content_types=['application/x-npy'],
    response_types=['application/json'],
    inference_instances=['ml.m5.large'],
    transform_instances=['ml.m5.large'],
    model_package_group_name='HiscoxClaimsModelGroup',
    approval_status='PendingManualApproval',
    description='Hiscox claims prediction model v1'
)

model_package.wait_for_registration()