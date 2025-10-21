from azure.ai.ml import MLClient, Input, command
from azure.ai.ml.entities import JobResourceConfiguration
from azure.identity import DefaultAzureCredential

# Configs
subscription_id = "07fd1fc4-9d9b-4bbe-b68a-dd42bf1df26e"
resource_group = "object_manipulation"
workspace_name = "CV-workspace"

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)

# Data
imagenet_input = {
    "data_path" : Input(
        type="uri_folder",
        path="azureml://datastores/cvdatastore/paths/ILSVRC2012/",
        mode="mount")
        }

# Command
'''
job = command(
    code=".",
    command=(
        "torchrun --nproc_per_node=4 "
        "train.py "
        "--data_path ${{inputs.data_path}} "
        "--base_dim 32 "
        "--epochs 5 "
        "--per_gpu_batch 8 "
        "--lr 5e-4 "
        "--num_workers 4 "
        "--output_dir ./outputs"
    ),
    inputs=imagenet_input,
    environment="azureml://registries/azureml/environments/AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu/versions/10",
    compute="NC64as-T4-V3-Cluster00",
    display_name="ImageNet-1k-dhvn-ddp-training",
    experiment_name="ImageNet-1k-DHVN-DDP-Experiment",
    description="Distributed training of DHVNClassification on 4xT4 GPUs using DDP and AMP for 5 epochs on ImageNet-1k 2012",
    resources=JobResourceConfiguration(
        instance_count=1,
        instance_type="Standard_NC64as_T4_v3"
    ),
)'''

job = command(
    code=".",
    command=(
        "torchrun --nproc_per_node=4 "
        "train.py "
        "--data_path ${{inputs.data_path}} "
        "--base_dim 32 "
        "--epochs 5 "
        "--per_gpu_batch 8 "
        "--lr 5e-4 "
        "--num_workers 4 "
        "--output_dir ./outputs"
    ),
    inputs=imagenet_input,
    environment="azureml://registries/azureml/environments/AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu/versions/10",
    compute="NC64as-T4-V3-Cluster01",
    display_name="test-ImageNet-10-subset-dhvn-ddp",
    experiment_name="test-DHVN-DDP-Experiment",
    description="Distributed training of DHVNClassification on 4xT4 GPUs using DDP and AMP for 5 epochs on ImageNet-1k 2012",
    resources=JobResourceConfiguration(
        instance_count=1,
        instance_type="Standard_NC64as_T4_v3"
    ),
)


# Submitting the job
return_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {return_job.name}")
print(f"Monitor at: {return_job.studio_url}")
