---
title: "Data Version Control"
date: 2023-09-10T23:29:21+08:00
draft: false
ShowToc: true
category: [ai]
tags: ["MLOps", "Version Control", "Experimentation"]
description: "Setting Up Data Version Control (DVC) Experiment Tracking Workflow for Your Machine Learning Project"
summary: "Setting Up Data Version Control (DVC) Experiment Tracking Workflow for Your Machine Learning Project"
---

# Data Version Control

## Setting Up Data Version Control (DVC) Experiment Tracking Workflow for Your Machine Learning Project

Data Version Control (DVC) is a valuable tool for managing data and code in machine learning projects. To enhance your workflow further, you can integrate DVC's built-in experiment tracking capabilities. In this comprehensive guide, we'll walk you through setting up DVC for a machine learning project and show you how to leverage DVC for experiment tracking. We'll use a hypothetical project of building a logistic regression model as an example.

## Project Structure

Let's start by organizing our project structure:

```
dvc-sample
├── data
│   ├── ground_truth.csv
│   └── features.csv
│
├── models
│   ├── logistic_regression_model.pkl
│   └── auc_scores.csv
│
├── src
│   ├── train_model.py
│   ├── predictions.py
│   └── utils.py
│
└── experiments
    ├── experiment_1
    ├── experiment_2
    └── experiment_3

```

In this structure, we have an `experiments` folder to track different experiment runs.

## Prerequisites

Before we begin, ensure you have DVC installed:

```bash
pip install dvc
```

## Initialize DVC

1. **Initialize DVC**: Begin by initializing DVC in your project directory.

```bash
dvc init
```

1. **Add Project Folders to DVC**: Specify which folders you want DVC to track.

```bash
dvc add data/ models/ src/
```

This command generates a `.dvc` file for each folder you selected, enabling DVC to monitor changes to the data and code.

## Adding .dvc Files to Git

To seamlessly integrate DVC with Git, include the generated `.dvc` files in your version control system (Git, in this case).

```bash
git add .gitignore data.dvc models.dvc src.dvc
```

By doing this, you connect your data and code with DVC for effective versioning control.

## Storing Data in a Remote Storage

Since large files like datasets and models should not be stored directly in your Git repository, we'll utilize a remote storage system. In this example, we'll continue to use Google Cloud Storage.

1. **Add Google Storage Bucket Path**: Inform DVC where to store your data in the cloud.

```bash
dvc remote add -d gs gs://your-bucket-name/your-project-name/
```

1. **Set Google Credentials**: Export your Google Cloud credentials JSON file.

```bash
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/credentials/key.json'
```

1. **Add DVC Configuration to Git**:

```bash
git add .dvc/config
```

## Commit and Push

Now, it's time to commit your changes and push your data to Google Cloud Storage.

```bash
git commit -m "Initialized DVC and added project files"
dvc push
```

These commands ensure your project configuration is committed, and your data is pushed to the remote storage.

## Experiment Tracking with DVC

DVC provides a convenient way to track experiments. Each experiment is treated as a separate branch of your DVC pipeline.

1. **Create a New Experiment Branch**:

```bash
dvc exp branch experiment_1
```

This command creates a new experiment branch named "experiment_1."

1. **Switch to the Experiment Branch**:

```bash
dvc exp switch experiment_1
```

Now, you are in the "experiment_1" branch, which is a separate workspace for your experiment.

1. **Run Your Experiment**:

Execute your machine learning code and experiments within this branch. Any changes you make here will only affect this experiment.

1. **Commit and Record Experiment Metrics**:

```bash
dvc exp run -n train_model \\
  -d src/train_model.py -d data/features.csv -d data/ground_truth.csv \\
  -o models/logistic_regression_model.pkl \\
  python src/train_model.py
```

This command not only commits your code changes but also records the metrics and dependencies for this specific experiment.

## Comparing Experiments

To compare different experiments and their results, you can switch between branches.

```bash
dvc exp switch main  # Switch back to the main branch
```

And then switch to another experiment branch:

```bash
dvc exp switch experiment_2  # Switch to another experiment branch
```

You can repeat this process to explore the results of each experiment.

## Using DVC and Experiment Tracking in a New Environment

If someone else wants to use your project with DVC and experiment tracking:

```bash
# Clone the Git repository
git clone <https://github.com/your-username/your-project.git>

# Navigate to the project directory
cd your-project

# Check available branches (experiment branches)
dvc exp show

# Switch to the experiment branch of interest
dvc exp switch experiment_1  # Replace with the desired experiment branch name

# Fetch data and code
dvc pull
```

With these steps, users can replicate your project's environment, access the same data and code, and explore different experiment branches.

## Generating Experiment Comparison Reports with DVC

One of the powerful features of Data Version Control (DVC) is its ability to not only track experiments but also generate comprehensive reports comparing the results of different experiments. In this section, we will show you how to leverage DVC to create experiment comparison reports in your machine learning project.

## Comparing Experiments

We have already set up DVC to manage our experiments within the `experiments` directory.

To compare different experiments and generate a report, follow these steps:

1. **Switch to the Main Branch**:
    
    Before generating a report, switch back to the main branch:
    
    ```bash
    dvc exp switch main
    ```
    
2. **Compare Experiments**:
    
    Use the `dvc exp diff` command to compare experiments and generate a report:
    
    ```bash
    dvc exp diff experiment_1 experiment_2 experiment_3 -o comparison_report.html
    ```
    
    This command compares the specified experiments (`experiment_1`, `experiment_2`, and `experiment_3`) and generates a report named `comparison_report.html`.
    
3. **View the Report**:
    
    You can view the generated report in your web browser. Simply open the HTML file:
    
    ```bash
    open comparison_report.html  # On macOS
    ```
    
    The report provides a detailed comparison of the specified experiments, including metrics, code changes, and data dependencies.
    

## Customizing the Report

DVC allows you to customize the report by specifying the information you want to include. You can choose to focus on specific metrics, code changes, or data dependencies based on your project's requirements.

Here's an example of customizing the report to include specific metrics:

```bash
dvc exp diff experiment_1 experiment_2 experiment_3 -m accuracy -o custom_report.html
```

In this command, we specify the `-m` flag followed by the metric name ("accuracy") to focus on that specific metric in the report.

## Using Reports for Decision-Making

Experiment comparison reports generated by DVC are invaluable for making informed decisions about model improvements, algorithm changes, or data preprocessing steps. These reports provide a clear overview of how different experiments perform and help you identify the most promising approaches.

## Conclusion

By following this guide, you can harness the full potential of Data Version Control (DVC) to not only track experiments but also generate insightful reports. These reports will significantly enhance your machine learning project's decision-making process, offering a powerful tool for data scientists and machine learning engineers to optimize models and achieve superior results.

DVC provides a robust solution for managing both data and code in your machine learning projects. When seamlessly integrated with DVC's built-in experiment tracking, you gain the capability to efficiently track, compare, and analyze diverse experiments. This integration streamlines collaboration and project management, transforming your workflow into a well-organized and efficient environment.

By implementing these practices, you'll establish a strong foundation for your machine learning endeavors, empowering you to make data-driven decisions and continually enhance your models for better performance and results.