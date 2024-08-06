# Model Development Strategies
<br><br>Growing my understanding of MLOps has meant explicitly structuring my workflows in solving a given problem, to optimize chances of successful model experiments. So, I developed this workflow to help ensure consistency in the quality of my developed models.<br><br><br>

## Personal Workflow<br><br>
<img src="workflow.png" alt="workflow" width="860" height="780"/><br><br><br>
*Model development workflow**<br><br><br>

This is what I typically use to develop models, as I can use open-source tooling to implement it.<br><br>

However, cloud technologies (AWS, Azure, or GCP) are a better choice for production, as they can help automate most aspects of this workflow for greater efficiency and model performance. Therefore, I am constantly growing my cloud engineering skills to be able to architect end-to-end ML solutions in this way instead, as that provides a more robust solution for MLOps.<br><br>

Here is a general overview of MLOps more suitable for production models:<br><br>

## General<br><br>
<img src="mlops.svg" alt="mlops" width="920" height="580"/><br><br><br>
*MLOps schematic: https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning*<br><br>

This would be a more robust pipeline for development, deployment and monitoring, including continous training capabilities to further the level of MLOps maturity.<br><br>

The following are illustrations of specific cloud solution architectures for highly mature MLOps processes:<br><br>

## <img src="azure_logo.png" style="padding: 10px 40px 0px 0px;" height="39"/> Azure<br><br>

<img src="azure_mlops.png" alt="azure mlops" width="920" height="580"/><br><br><br>

*Azure MLOps architecture: https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/mlops-maturity-model-with-azure-machine-learning/ba-p/3520625*<br><br><br>


## <img src="aws_logo.png" style="padding: 10px 40px 0px 0px;" height="39"/> AWS<br><br>

<img src="aws_mlops.jpg" alt="aws mlops" width="920" height="590"/><br><br><br>

*AWS MLOps architecture: https://aws.amazon.com/blogs/machine-learning/automate-model-retraining-with-amazon-sagemaker-pipelines-when-drift-is-detected/*<br><br><br>


## <img src="gcp_logo.png" style="padding: 10px 40px 0px 0px;" height="39"/> GCP<br><br>

<img src="gcp_mlops.svg" alt="gcp mlops" width="920" height="590"/><br><br><br>

*GCP MLOps architecture: https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build*<br><br><br>

My goal is to apply cloud solution architectures to proposed problems through AWS, Azure or GCP to continuously evolve the maturity of my ML workflows.