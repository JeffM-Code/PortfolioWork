# Model Development Strategies
<br><br>Growing my understanding of MLOps has meant explicitly structuring my workflows in solving a given problem, to optimize chances of successful model experiments. So, I developed this workflow to help ensure consistency in the quality of my developed models.<br><br><br>

## Personal Workflow<br><br>
<img src="workflow.png" alt="workflow" width="940" height="780"/><br><br><br>
*Model development workflow**<br><br><br>

This is what I typically use to develop models, as I can use open-source tooling to implement it.<br><br>

# Motivation
## *Problem definition & Scoping*:
At this stage, the problem is selected based on what I believe to be an interesting problem, whose resulting inference and predicitons I can interpret well, i.e., energy / physics problems.<br><br>

# Code
Technology stack inlcudes mainly open-source software for engineering the data and developing the model.


# Data
I normally like to use Kaggle, Huggingface or a company / business website with good, high quality data (like Elexon).<br><br>


## *Data Ingestion*:
Some steps are done manually, but in future will be scraped or interacted with programmatically through APIs.

## *Data Preparation*:
This is done after analysis of the data programmatically, to see fit with intended model.

# Model
## *Feature engineering*:
Feature Transformation:<br>
Including scaling, log transformations, encoding, and so on, to improve performance and interpretiability.<br><br>

Dimensionality Reduction / Feature Extraction:<br>
To reduce the number of features or "dimentionality" of the feature space, which can be done through extracting some number of features from a larger space such as through PCA or LDA.<br><br>

Feature Selection:<br>
Methods for selection of relevant features include filter, wrapper, or embedded methods to ensure appropriate features used for model training.<br><br>


### Metrics
a. Performance:<br>
Accuracy of model prediction should be reasonable.<br><br>

b. Runtime:<br>
They need to make predictions in a reasonable amount of time, some models i've made do take some time with more data made available for prediction, which is something that can be improved with a good feedback loop or iteration of model development.<br><br>

c. Interpretiability:<br>
Model outputs should be well interepretable, so since I work in domains I am experienced with (physics, energy) I can guide the reader towards what a sensible prediction looks like.<br><br>


d. Generalizability:<br>
They should work well on unseen data, which I emulate through holdout sets to test this for each model.<br><br>

## *Modelling*:
Model algorithm is selected, trained, and tuned appropriately here, using the various methods described above including feature engineering.<br>

Model selection techniques are highly specific depending on the problem, so these are done based on the needs of satisfying appropriate solution to problem proposed.<br><br>

## *Deployment*:
This is normally done on Streamlit, as it's a free platform to host models, and as models developed are normally small in size within their containers, it seems appropriate to host them there.<br><br>

## *Feedback Loop*:
This involves the monitoring of deployed models and looping back to data preparation or problem definition depending on severity of performance issues observed.<br><br>

# Production
In general, cloud technologies (AWS, Azure, or GCP) are a better choice for true, production models, as they can automate most aspects of the workflow for greater efficiency and model performance.<br><br>

Therefore, I am constantly growing my cloud engineering skills to be able to architect end-to-end ML solutions in this way instead, as that provides a more robust solution for MLOps.<br><br>

Here is a general overview of MLOps more suitable for production models:<br><br>
<img src="mlops.svg" alt="mlops" width="920" height="580"/><br><br><br>
*MLOps schematic: https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning*<br><br>

This would be a more robust pipeline for development, deployment and monitoring, including continous training capabilities to further the level of MLOps maturity.<br><br>

The following are illustrations of specific cloud solution architectures for highly mature MLOps processes:<br><br>

## <img src="azure_logo.png" style="padding: 10px 40px 0px 0px;" height="39"/> Azure<br><br>

<img src="azure_mlops.png" alt="azure mlops" width="970" height="580"/><br><br><br>

*Azure MLOps architecture: https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/mlops-maturity-model-with-azure-machine-learning/ba-p/3520625*<br><br><br>


## <img src="aws_logo.png" style="padding: 10px 40px 0px 0px;" height="39"/> AWS<br><br>

<img src="aws_mlops.jpg" alt="aws mlops" width="920" height="590"/><br><br><br>

*AWS MLOps architecture: https://aws.amazon.com/blogs/machine-learning/automate-model-retraining-with-amazon-sagemaker-pipelines-when-drift-is-detected/*<br><br><br>


## <img src="gcp_logo.png" style="padding: 10px 40px 0px 0px;" height="39"/> GCP<br><br>

<img src="gcp_mlops.svg" alt="gcp mlops" width="970" height="590"/><br><br><br>

*GCP MLOps architecture: https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build*<br><br><br>

My goal is to apply cloud solution architectures to proposed problems through AWS, Azure or GCP to continuously evolve the maturity of my ML workflows.