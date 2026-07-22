# Breast cancer risk prediction from longitudinal mammography reports in a real-world health system
 
 
# Introduction
Here we provide the pipeline for: [Breast cancer risk prediction from longitudinal mammography reports in a real-world health system](to-be-released)

### Citation

> XXXXJOURNALXXXX. DOI: XXXX


# Abstract

Breast cancer screening protocols apply uniform strategies across populations with vastly different underlying risk profiles, limiting early detection in high-risk individuals and generating unnecessary burden in low-risk ones. Risk-based screening has been proposed as a strategy to tailor screening intensity to individual risk, but its implementation requires accurate, calibrated, and scalable risk models validated in real-world settings. Here we develop and validate a population-scale breast cancer risk model using longitudinal mammography reports and clinical data from approximately 4.6 million examinations of 1.8 million patients across the largest private health system in Brazil. Without requiring mammography images, the model achieves \textcolor{blue}{strong and clinically meaningful} discrimination for predicting breast cancer within one to five years, with a 5-year AUROC of 0.86 (95\% CI: 0.85--0.87) and consistent performance across shorter horizons. Flagging the top 3.4\% highest-risk exams identifies 50.1\% of future cancer cases, enabling earlier detection, while in the bottom 47\% risk percentile only 10.0\% develop cancer within five years, suggesting approximately half of all patients could safely transition to longer screening intervals.  An explainability analysis reveals distinct and clinically meaningful patterns driving predictions at both ends of the risk spectrum. These results establish the feasibility of interpretable, image-free, population-scale risk modeling as a practical pathway toward personalized breast cancer screening.

__Fig. X XXXX.__
*(a) Epidemiological curves of Covid-19 cases and deaths in Fortaleza, and major interventions implemented in the city, Jan 2020-July 2022. (b) Cases (gray) and (c) deaths (red) of Covid-19 and cumulative distribution of individuals vaccinated with 2 doses (we consider a single dose of the Janssen vaccine as equivalent to a two-dose regimen).* 



## Overview

This repository implements an end-to-end data and modeling pipeline for breast cancer risk prediction using text-based and structured healthcare data provided by the health insurer/provider Hapvida in Brazil.

The modeling objective is:

> Estimate the probability of developing breast cancer within a defined prediction horizon, conditional on information available at the time of a benign mammogram exam.

The intended users of this software are data scientists developing, validating, and maintaining breast cancer risk models for prospective patient monitoring and screening optimization.

The pipeline performs the following stages:

1. **Data extraction**:
   - Oracle SQL queries over databases.
   - Schema validation of extracted tables.

2. **Preprocessing**:
   - Preprocessing of raw data.
   - Merging of preprocessed sources.

3. **Cohort construction**:
   - Extra processing.
   - Index date definition.
   - Inclusion and eligibility criteria.
   - Survival calculations.

4. **Precomputation**:
   - Aggregation of past mammogram reports.
   - Temporal feature construction.

5. **Cohort splitting**

6. **Model development**:
   - Hyperparameter tuning.
   - Training the best parameter configuration.

7. **Evaluation**:
   - Held-out test metrics (AUROC and AUPRC).
   - Explainability calculations (integrated gradients and feature ablation).

---

## Quickstart

All execution steps should be performed through the CLI interface (`hapcancer.cli`) to ensure consistent configuration and logging. Scripts for direct interaction with the CLI interface were created for defining and automating major pipeline tasks.

All commands below must be executed from the repository root.

### 1. Create and Activate Environment

    conda env create -f environment.yml
    conda activate hapcancer

### 2. Install Package in Editable Mode (not optional for running the scripts)

    pip install -e .

This mode allows the user to call the package directly from the CLI, using commands such as `hapcancer command [args]`.

**Examples of how to run CLI tasks**

Option 1 (if editable mode is not enabled):

    python -m hapcancer.cli command-name [args]

Option 2 (editable mode enabled):

    hapcancer command-name [args]

Option 1 is not recommended for automation. Further details on running the scripts are provided in the `docs` folder.

---

## Repository Structure

    hapcancer/
      etl/           # Querying, preprocessing, merging, and cohort definitions
      schemas/       # Data contracts and validation
      model/         # Architecture, training, losses
      eval/          # Metrics and evaluation
      cli/           # Command-line entrypoints for running pipeline stages

---

## Reproducibility

Every major experiment (defined by running all pipeline stages) should be reproducible and traceable to a configuration folder. This software can only be executed through the correct setup of a configuration folder. More details on how to set up a configuration folder are provided in `docs/configuration_setup.md`. Here, we give a short overview.

A configuration folder follows a specific schema and contains several `.yml` configuration files on which each task depends. Every minor and major component of the pipeline uses information stored in these configuration files.

Examples include:

1. File system paths to store extracted raw data.
2. Main parameters for the BI-RADS classifier to be trained.
3. Age intervals for which a patient (at the time of an exam) is considered eligible.
4. BI-RADS values to include in the initial cohort definition.
5. ...

Refer to `docs/configuration_setup.md` for more details.

---


# Correspondence
For any issues with anonymization or major issues with the functionality of the script please [create an issue](https://github.com/higorsmonteiro/breastaicancer/issues).


## License
The data collected and presented is licensed under the [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/), and the underlying code used to format, analyze and display that content is licensed under the [MIT license](http://opensource.org/licenses/mit-license.php). 


# Authors
- __Higor S Monteiro__: Department of Physics, Universidade Federal do Ceará | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [GitHub Profile](https://github.com/higorsmonteiro)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Google Scholar](xxxx)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Profile](xxxx)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Google Scholar](xxxx)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Google Scholar](xxxx)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Google Scholar](xxxx)
- __XXXX__: XXXX | ![link--v2](https://user-images.githubusercontent.com/43140693/111211993-742aeb80-85a5-11eb-85b8-a1e2c5102d99.png) : [Google Scholar](xxxx)
