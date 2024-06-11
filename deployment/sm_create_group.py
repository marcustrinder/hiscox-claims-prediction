import boto3

sm_client = boto3.client('sagemaker')

response = sm_client.create_model_package_group(
    ModelPackageGroupName='HiscoxClaimsModelGroup',
    ModelPackageGroupDescription='Model package group for Hiscox claims prediction'
)