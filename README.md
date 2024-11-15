# Thesis Findings

In this github repo, you will find the data regarding Experiment 1, Experiment 2 and the simulation code.

## Experiment 1
In this folder, you will find the cleaned-dataset that was used for data analysis, and the Qualtrics form that was used to collected the data.

## Experiment 2
In this folder, you will find the cleaned-dataset that was used for data analysis, the Qualtrics form that was used to collected the data, and the different types of interventions used.

## Simulation Study 
In this folder, you will find the model_selection.py which denotes the function that will compare the different models and output the best model based on AUC.
Multiple-Configuration-Simulation.py - This Python script simulates and compares the effectiveness of personalized interventions versus baseline interventions in reducing users’ susceptibility to misinformation. It utilizes synthetic data for user profiles and content items, simulates user-content interactions, applies machine learning models to predict susceptibility, and evaluates intervention strategies over multiple time steps.

Key Features:

	1.	Data Simulation:
	•	Generates synthetic user profiles and content items with diverse attributes.
	•	Simulates interactions between users and content over time.
	2.	Intervention Strategies:
	•	Personalized Interventions: Tailored to individual user profiles and content characteristics.
	•	Baseline Interventions: Generic strategies applied randomly.
	3.	Effectiveness Evaluation:
	•	Evaluates the success of interventions based on parameters like content complexity, emotional impact, and user attributes.
	•	Tracks changes in user susceptibility scores over time.
	4.	Machine Learning:
	•	Predicts user susceptibility using features from interactions.
	•	Applies a selected model (via select_best_model).
	5.	Parameter Optimization:
	•	Allows experimentation with different parameter settings (e.g., susceptibility decay, intervention effectiveness, content weighting).
	6.	Visualization:
	•	Compares cumulative effectiveness of personalized and baseline systems.
	•	Tracks average user susceptibility and effectiveness over time using plots.
	7.	Results Storage:
	•	Saves simulation results to a CSV file for further analysis.

## Systematic Literature Review
This excel contains the steps that were used to gather the papers used in the SLR.s