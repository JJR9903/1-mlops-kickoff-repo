# Telecom Customer Churn Prediction – MLOps Golden Repository 

 Author: MBDS2026 – MLOps Engineering Group 4: Rincon Juan José, Gelin Romain, Gaitan Diego, Tcheishvili Luka, Tambey Cécile 
 Course: Machine Learning Operations V4.0.0 
 Program: MSc in Business Analytics and Data Science 
 Institution: IE University 
 Academic Year: 2026 

 ---

 ## 1. Business Objective

 ### Strategic Goal

 The objective of this project is to design, implement, and validate a production-oriented Machine Learning Operations (MLOps) pipeline capable of predicting customer churn in a telecommunications environment. The system is intended to transition churn modeling from exploratory notebook-based experimentation to a structured, deterministic, and auditable operational workflow.

 In subscription-based industries such as telecommunications, churn directly impacts recurring revenue and customer lifetime value. Acquisition costs significantly exceed retention costs; therefore, even marginal reductions in churn generate disproportionately large financial benefits. This project operates churn prediction as a decision-support tool for proactive retention strategies.

 ### Target Stakeholders

 The primary users of this solution are the Customer Retention & Revenue Management teams. These stakeholders require structured churn probability scores to prioritize outreach, optimize retention incentives, and improve campaign return on investment.

 ---

 ## 2. Success Metrics

 ### Business KPIs

 The intended business impact is a measurable reduction in preventable churn through targeted retention strategies. A realistic performance benchmark for deployment would target a 3–5% relative reduction in churn rate compared to heuristic-based targeting approaches. Secondary business benefits include improved allocation efficiency of retention budgets and stabilization of recurring revenue streams.

 ### Technical KPIs

 The primary evaluation metric is the F1 score for the churn class, selected due to class imbalance and the need to balance precision and recall. ROC-AUC is used as a secondary ranking metric to assess discrimination performance independently of threshold selection.

 The solution must outperform a baseline classifier and execute deterministically from a single-entry point while passing all unit tests.

 ---

 ## 3. Business Context and Maturity Assessment

 The client is assumed to be a large telecommunications operator operating in a highly competitive, price-sensitive market with low switching barriers.

 The organization demonstrates moderate data maturity, as structured customer data is available. However, operational ML maturity remains limited due to reliance on manual notebook workflows, lack of standardized validation gates, and absence of automated retraining processes.

 Tooling maturity is sufficient for experimentation but not fully production ready. Process maturity lacks strict reproducibility and artifact traceability. From a strategic perspective, churn reduction is a priority, but execution mechanisms require automation and governance improvements.

 ---

 ## 4. Problem Statement

 Churn detection is currently reactive and based on retrospective analysis or heuristic rules. This results in inefficient retention campaigns and suboptimal allocation of marketing resources.

 Without predictive ranking, retention actions may target low-risk customers while missing high-risk customers. Additionally, notebook-based experimentation introduces reproducibility risk, making it difficult to operate insights consistently.

 ---

 ## 5. Proposed Solution and Key Functionalities

 This repository implements a fully modular MLOps pipeline that includes:

 - Deterministic data ingestion and cleaning 
 - Schema validation and fail-fast quality gates 
 - Leakage-safe feature engineering via ColumnTransformer 
 - Unified model training through a Scikit-Learn Pipeline 
 - Structured evaluation using class-sensitive metrics 
 - Automated unit testing for module reliability 
 - Artifact generation for reproducibility 

 The solution is executed from a single-entry point:

 	python -m src.main

 This architecture prevents hidden state, enforces separation of concerns, and ensures consistent behavior across environments.

 ---

 ## 6. Data Description

 Dataset: Customer Churn in Telecom Sample Dataset by IBM 
 Source: https://www.kaggle.com/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm 

 The dataset contains 7,043 customer records with 21 features, including demographic information, service subscriptions, contract types, billing attributes, and churn status.

 The target variable represents binary churn classification. Raw data is excluded from version control to comply with data governance best practices.

 ---

 ## 7. Methodology

 ### Data Processing

 Data cleaning includes normalization of column naming, missing value handling, deterministic indexing, and explicit target validation.

 Validation gates ensure that schema contracts are respected before proceeding to training, reducing silent failure risk.

 ### Feature Engineering

 Feature engineering is implemented using a Scikit-Learn ColumnTransformer blueprint that is returned unfitted. This ensures that preprocessing is fit strictly on the training split, preventing leakage.

 Domain-driven features include tenure-based churn risk segmentation and service subscription intensity indicators.

 ### Modeling Approach

 A unified Scikit-Learn Pipeline bundles preprocessing and model training. Deterministic random states are applied to ensure reproducibility.

 ### Evaluation

 Performance is evaluated using F1 score and ROC-AUC on held-out validation/test splits only. No transformation is fit on non-training data.

 ---

 ## 8. Testing Strategy

 Each core module includes unit tests that validate:

 • Correct function contracts 
 • Schema validation behavior 
 • Feature preprocessing construction 
 • Safe model fitting 
 • Artifact creation 

 All tests are executed via: pytest

 This test suit ensures integration stability and protects against regression during collaborative development.

 ---

 ## 9. Risk Analysis and Mitigation

 Potential risks include model drift, schema drift, and leakage during preprocessing. These are mitigated through deterministic splitting, validation gates, and pipeline-based transformation logic.

 Operational risks associated with manual notebook workflows are mitigated by enforcing a single-entry point, structured artifact storage, and automated testing.

 ---

 ## 10. Cost and Timeline Estimate

 A minimal viable production-ready version of this solution can be delivered within approximately six to eight weeks under a focused development scope.

 An enterprise-grade rollout including monitoring, CI/CD automation, and governance integration may extend to approximately twelve weeks.

 Estimated resource allocation includes a Data Scientist, MLOps Engineer, Data Engineer, and domain Subject Matter Expert.

 ---

 ## 11. Repository Structure
 .
 .
├── README.md            	# Project definition and documentation
├── environment.yml      	# Dependency management (Conda)
├── config.yaml          	# Central configuration (paths, parameters)
├── .env                 	# Secrets placeholder (not committed)
│
├── notebooks/           	# Experimental sandbox
│   └── baseline_experiments.ipynb
│
├── src/                 	# Production code (Modular pipeline)
│   ├── __init__.py      	# Python package definition
│   ├── load_data.py     	# Raw data ingestion
│   ├── clean_data.py    	# Data preprocessing and cleaning
│   ├── validate.py      	# Data quality and schema validation
│   ├── features.py      	# Feature engineering blueprint
│   ├── train.py         	# Model training and artifact creation
│   ├── evaluate.py      	# Performance evaluation
│   ├── infer.py         	# Inference logic
│   └── main.py          	# Pipeline orchestrator
│
├── data/                	# Local storage (ignored by Git)
│   ├── raw/             	# Immutable input data
│   └── processed/       	# Cleaned data ready for training
│
├── models/              	# Serialized model artifacts (ignored by Git)
│
├── reports/             	# Generated metrics and prediction artifacts
│
└── tests/               	# Automated unit tests
 This structure enforces separation between experimentation and production code, aligning with MLOps modularization principles.

 ---

 ## 12. Reproducibility and Environment Management

 Dependencies are defined in environment.yml to ensure environment discipline and prevent “works on my machine” failures.

 Environment setup:

 	conda env create -f environment.yml
 	conda activate mlops_project

 Pipeline execution:

 	python -m src.main

 Unit testing:

 	pytest

 Artifacts generated:

 • data/processed/clean.csv 
 • models/model.joblib 
 • reports/predictions.csv 

 ---

 ## 13. Scalability Considerations

 The modular design supports extension to additional predictive use cases, integration into CI/CD pipelines, automated retraining schedules, and API-based deployment.

 This implementation aligns with Level 1 MLOps maturity, enabling pipeline automation and forming a foundation for continuous deployment.

 ---

 ## 14. Conclusion

 This project demonstrates the systematic transformation of exploratory churn modeling into a robust, modular, and production-oriented MLOps pipeline.

 The solution integrates business strategy with engineering discipline by enforcing determinism, validation, testing, and artifact traceability. It establishes a scalable framework for AI-driven churn reduction initiatives and reflects adherence to professional MLOps best practices.
  

