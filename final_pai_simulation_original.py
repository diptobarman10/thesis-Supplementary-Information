import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
import random
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import logging

# Optional: For visualization
import matplotlib.pyplot as plt
import seaborn as sns  # Added for advanced plotting

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Add after imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def safe_divide(numerator, denominator):
    """Safely perform division with error handling"""
    try:
        return numerator / denominator if denominator != 0 else 0
    except Exception as e:
        logging.error(f"Division error: {e}")
        return 0

# Update the effectiveness calculation in simulation functions
def calculate_effectiveness(effective, total):
    """Calculate effectiveness with error handling"""
    try:
        return safe_divide(effective, total) * 100
    except Exception as e:
        logging.error(f"Error calculating effectiveness: {e}")
        return 0

# -------------------------------
# Data Generation Functions
# -------------------------------

class UserProfile:
    """Class to manage user profiles and their attributes"""
    def __init__(self, num_users):
        self.profiles = self._generate_profiles(num_users)
    
    def _generate_profiles(self, num_users):
        """Generates synthetic user profiles."""
        return pd.DataFrame({
            'user_id': range(num_users),
            'CRT_score': np.random.randint(0, 8, num_users),
            'CMQ_score': np.random.randint(0, 8, num_users),
            'openness': np.random.rand(num_users),
            'conscientiousness': np.random.rand(num_users),
            'extraversion': np.random.rand(num_users),
            'agreeableness': np.random.rand(num_users),
            'neuroticism': np.random.rand(num_users),
            'political_ideology': np.random.choice(['liberal', 'moderate', 'conservative'], num_users),
            'age': np.random.randint(18, 65, num_users),
            'gender': np.random.choice(['male', 'female', 'other'], num_users),
            'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], num_users),
            'past_interventions': np.zeros(num_users),
            'susceptibility_score': np.random.rand(num_users)
        })
    
    def update_susceptibility(self, user_ids, change):
        """Update susceptibility scores for given users"""
        self.profiles.loc[self.profiles['user_id'].isin(user_ids), 'susceptibility_score'] += change
        # Ensure scores stay between 0 and 1
        self.profiles['susceptibility_score'] = self.profiles['susceptibility_score'].clip(0, 1)

class InterventionSystem:
    """Class to manage intervention selection and effectiveness"""
    def __init__(self, is_personalized=True):
        self.is_personalized = is_personalized
        self.interventions = {
            'Standard Warning': 0.5,
            'Informational Message': 0.5,
            'Neutral Feedback': 0.5,
            'Engagement Prompt': 0.5,
            'Prebunking (Context)': 0.8,
            'Boosting (Educational Video)': 0.8,
            'Nudge Warning': 0.8
        }
    
    def select_intervention(self, row):
        """Select appropriate intervention based on system type"""
        if self.is_personalized:
            return self._select_personalized(row)
        return self._select_baseline(row)
    
    def _select_personalized(self, row):
        """Select intervention based on personalized criteria"""
        if row['political_ideology'] == 'conservative':
            return 'Nudge Warning'
        if row['CMQ_score'] > 4 and row['topic'] == 'politics':
            return 'Prebunking (Context)'
        elif row['CRT_score'] < 4 and row['complexity'] > 0.5:
            return 'Boosting (Educational Video)'
        else:
            return 'Standard Warning'
    
    def _select_baseline(self, row):
        """Randomly select from baseline interventions"""
        return random.choice(list(self.interventions.keys()))

# -------------------------------
# Interaction Simulation
# -------------------------------

def simulate_interactions(user_profiles, content_items, time_steps):
    """Simulates user-content interactions over multiple time steps."""
    interactions_list = []
    for t in range(time_steps):
        # Ensure susceptibility_score stays between 0 and 1
        user_profiles['susceptibility_score'] = user_profiles['susceptibility_score']
        # Merge user profiles and content items to simulate interactions
        interactions = pd.merge(user_profiles.assign(key=1), content_items.assign(key=1), on='key').drop('key', axis=1)
        # Create a synthetic target variable: 1 if user is susceptible, 0 otherwise
        interactions['susceptible'] = interactions.apply(lambda row: 
            int(
                row['is_misinformation'] == 1 and
                row['susceptibility_score'] > np.random.rand()
            ), axis=1)
        # Add time step
        interactions['time_step'] = t
        interactions_list.append(interactions)
    # Concatenate all interactions
    all_interactions = pd.concat(interactions_list, ignore_index=True)
    return all_interactions

# -------------------------------
# Feature Encoding
# -------------------------------

def encode_features(df):
    """Encodes categorical features using OneHotEncoder."""
    categorical_cols = ['political_ideology', 'gender', 'education', 'topic', 'sentiment']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.drop(categorical_cols, axis=1).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return df

# -------------------------------
# Model Training
# -------------------------------

def train_model(interactions):
    """Trains an XGBoost classifier to predict susceptibility."""
    # Encode features
    interactions_encoded = encode_features(interactions)
    # Define features and target - exclude intervention column if it exists
    columns_to_drop = ['user_id', 'content_id', 'susceptible', 'is_misinformation', 
                      'time_step', 'intervention']
    feature_columns = [col for col in interactions_encoded.columns 
                      if col not in columns_to_drop]
    X = interactions_encoded[feature_columns]
    y = interactions_encoded['susceptible']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model, feature_columns  # Return both model and feature columns

# -------------------------------
# Intervention Selection Functions
# -------------------------------

# Define baseline interventions and their selection probabilities
baseline_interventions = [
    'Standard Warning',
    'Informational Message',
    'Neutral Feedback',
    'Engagement Prompt'
]

# Define selection probabilities (they should sum to 1)
baseline_intervention_probs = [0.25, 0.25, 0.25, 0.25]  # Example probabilities

def select_intervention_personalized(row):
    """Selects intervention based on user and content features."""
    # New condition for political ideology towards conservative
    if row['political_ideology'] == 'conservative':
        return 'Nudge Warning'  # Assign 'Nudge Warning' to conservative users
    # Existing personalized intervention logic
    if row['CMQ_score'] > 4 and row['topic'] == 'politics':
        return 'Prebunking (Context)'
    elif row['CRT_score'] < 4 and row['complexity'] > 0.5:
        return 'Boosting (Educational Video)'
    else:
        return 'Standard Warning'

def select_intervention_baseline(row):
    """Randomly selects a baseline intervention for each interaction."""
    return random.choices(baseline_interventions, weights=baseline_intervention_probs, k=1)[0]

# -------------------------------
# Intervention Effectiveness Functions
# -------------------------------

# Define base effectiveness for each baseline intervention
baseline_effectiveness = {
    'Standard Warning': 0.5,          # 50% base effectiveness
    'Informational Message': 0.5,     # 40% base effectiveness
    'Neutral Feedback': 0.5,          # 30% base effectiveness
    'Engagement Prompt': 0.5          # 60% base effectiveness
}

def intervention_effectiveness_personalized(row):
    """Determines the effectiveness of personalized interventions."""
    if row['susceptible'] == 1:
        base_effectiveness = 0.0
        if row['intervention'] == 'Prebunking (Context)':
            base_effectiveness = 0.8 if row['CMQ_score'] > 4 and row['topic'] == 'politics' else 0.5
        elif row['intervention'] == 'Boosting (Educational Video)':
            base_effectiveness = 0.8 if row['CRT_score'] < 4 else 0.5
        elif row['intervention'] == 'Standard Warning':
            base_effectiveness = 0.5
        elif row['intervention'] == 'Nudge Warning':  # Handle 'Nudge Warning'
            base_effectiveness = 0.8  # Define base effectiveness for 'Nudge Warning'
        # Adjust effectiveness based on content complexity and emotional impact
        content_factor = (1 - row['complexity']) * 0.5 + (1 - row['emotional_impact']) * 0.5
        effectiveness = base_effectiveness * content_factor
        return int(random.random() < effectiveness)
    else:
        return 0  # Not susceptible, so intervention not needed

def intervention_effectiveness_baseline(row):
    """Determines the effectiveness of baseline interventions."""
    if row['susceptible'] == 1:
        # Get the base effectiveness based on the intervention type
        base_effectiveness = baseline_effectiveness.get(row['intervention'], 0.5)  # Default to 0.5 if not found
        
        # Adjust effectiveness based on content complexity and emotional impact
        content_factor = (1 - row['complexity']) * 0.5 + (1 - row['emotional_impact']) * 0.5
        effectiveness = base_effectiveness * content_factor
        
        # Determine if the intervention is effective
        return int(random.random() < effectiveness)
    else:
        return 0  # Not susceptible, so intervention not needed

# -------------------------------
# Intervention Simulation Functions
# -------------------------------

def simulate_interventions_over_time_personalized(interactions, model_tuple, user_profiles):
    """Simulates personalized interventions and updates user profiles."""
    model, feature_columns = model_tuple  # Unpack the tuple
    interactions_encoded = encode_features(interactions)
    X = interactions_encoded[feature_columns]  # Use only the selected feature columns
    interactions['predicted_susceptibility'] = model.predict_proba(X)[:,1]
    # Select interventions (Personalized)
    interactions['intervention'] = interactions.apply(select_intervention_personalized, axis=1)
    
    # Simulate user response
    interactions['effective'] = interactions.apply(intervention_effectiveness_personalized, axis=1)
    
    # Update user profiles based on effective interventions
    effective_interventions = interactions[interactions['effective'] == 1]
    users_with_effective_interventions = effective_interventions['user_id'].unique()
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'past_interventions'] += 1
    
    # Calculate effectiveness
    total_susceptible = interactions['susceptible'].sum()
    total_effective = interactions['effective'].sum()
    effectiveness_percentage = (total_effective / total_susceptible * 100) if total_susceptible > 0 else 0
    print(f"Personalized Intervention Effectiveness at time step {interactions['time_step'].iloc[0]+1}: {total_effective}/{total_susceptible} ({effectiveness_percentage:.2f}%)")
    
    return interactions, user_profiles, total_effective, total_susceptible

def simulate_interventions_over_time_baseline(interactions, model_tuple, user_profiles):
    """Simulates baseline interventions and updates user profiles."""
    model, feature_columns = model_tuple  # Unpack the tuple
    interactions_encoded = encode_features(interactions)
    X = interactions_encoded[feature_columns]  # Use only the selected feature columns
    interactions['predicted_susceptibility'] = model.predict_proba(X)[:,1]
    # Select interventions (Baseline)
    interactions['intervention'] = interactions.apply(select_intervention_baseline, axis=1)
    
    # Simulate user response
    interactions['effective'] = interactions.apply(intervention_effectiveness_baseline, axis=1)
    
    # Update user profiles based on effective interventions
    effective_interventions = interactions[interactions['effective'] == 1]
    users_with_effective_interventions = effective_interventions['user_id'].unique()
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'past_interventions'] += 1
    
    # Calculate effectiveness
    total_susceptible = interactions['susceptible'].sum()
    total_effective = interactions['effective'].sum()
    effectiveness_percentage = (total_effective / total_susceptible * 100) if total_susceptible > 0 else 0
    print(f"Baseline Intervention Effectiveness at time step {interactions['time_step'].iloc[0]+1}: {total_effective}/{total_susceptible} ({effectiveness_percentage:.2f}%)")
    
    return interactions, user_profiles, total_effective, total_susceptible

# -------------------------------
# Main Comparison Function
# -------------------------------

class SimulationConfig:
    """Configuration class for simulation parameters"""
    def __init__(self):
        self.num_users = 100
        self.num_content = 100
        self.time_steps = 10
        self.random_seed = 42
        self.test_size = 0.2
        self.learning_rate = 0.1  # New parameter for learning effect
        
        # Intervention effectiveness parameters
        self.base_effectiveness = {
            'Standard Warning': 0.5,
            'Informational Message': 0.5,
            'Neutral Feedback': 0.5,
            'Engagement Prompt': 0.5
        }

class PerformanceMetrics:
    """Class to track and analyze simulation performance"""
    def __init__(self):
        self.metrics = {
            'effectiveness_rates': [],
            'susceptibility_changes': [],
            'intervention_counts': {},
            'topic_effectiveness': {},
            'user_learning_rates': []
        }
    
    def update_metrics(self, interactions, user_profiles):
        """Update performance metrics after each time step"""
        self.metrics['effectiveness_rates'].append(
            safe_divide(
                interactions['effective'].sum(),
                interactions['susceptible'].sum()
            )
        )
        
        self.metrics['susceptibility_changes'].append(
            user_profiles['susceptibility_score'].mean()
        )
        
        # Update intervention counts
        intervention_counts = interactions['intervention'].value_counts()
        for intervention, count in intervention_counts.items():
            self.metrics['intervention_counts'][intervention] = \
                self.metrics['intervention_counts'].get(intervention, 0) + count
    
    def generate_report(self):
        """Generate a comprehensive performance report"""
        report = {
            'average_effectiveness': np.mean(self.metrics['effectiveness_rates']),
            'susceptibility_reduction': self.metrics['susceptibility_changes'][0] - 
                                     self.metrics['susceptibility_changes'][-1],
            'most_used_intervention': max(self.metrics['intervention_counts'].items(), 
                                        key=lambda x: x[1])[0] if self.metrics['intervention_counts'] else None
        }
        return report

def main_comparison():
    # Initialize configuration
    config = SimulationConfig()
    
    # Set up logging
    logging.info("Starting simulation comparison")
    
    try:
        # Generate user profiles
        user_profiles = UserProfile(config.num_users).profiles
        
        # Generate content items
        content_items = pd.DataFrame({
            'content_id': range(config.num_content),
            'topic': np.random.choice(['politics', 'health', 'science', 'technology'], config.num_content),
            'complexity': np.random.rand(config.num_content),
            'emotional_impact': np.random.rand(config.num_content),
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], config.num_content),
            'is_misinformation': np.random.choice([0, 1], config.num_content, p=[0.7, 0.3])
        })
        
        # Initialize systems
        personalized_system = InterventionSystem(is_personalized=True)
        baseline_system = InterventionSystem(is_personalized=False)
        
        # Initialize metrics
        personalized_metrics = PerformanceMetrics()
        baseline_metrics = PerformanceMetrics()
        
        # Run simulations for each time step
        for t in range(config.time_steps):
            # Simulate interactions
            interactions = simulate_interactions(user_profiles, content_items, 1)
            
            # Train model on current interactions
            model_tuple = train_model(interactions)
            
            # Run personalized interventions
            personalized_results, user_profiles_p, total_effective_p, total_susceptible_p = \
                simulate_interventions_over_time_personalized(interactions, model_tuple, user_profiles.copy())
            
            # Update personalized metrics
            personalized_metrics.update_metrics(personalized_results, user_profiles_p)
            
            # Run baseline interventions
            baseline_results, user_profiles_b, total_effective_b, total_susceptible_b = \
                simulate_interventions_over_time_baseline(interactions, model_tuple, user_profiles.copy())
            
            # Update baseline metrics
            baseline_metrics.update_metrics(baseline_results, user_profiles_b)
        
        # Generate final reports
        personalized_report = personalized_metrics.generate_report()
        baseline_report = baseline_metrics.generate_report()
        
        # Print comparison results
        print("\nFinal Results:")
        print("Personalized System:")
        print(f"Average Effectiveness: {personalized_report['average_effectiveness']:.2f}")
        print(f"Susceptibility Reduction: {personalized_report['susceptibility_reduction']:.2f}")
        print(f"Most Used Intervention: {personalized_report['most_used_intervention']}")
        
        print("\nBaseline System:")
        print(f"Average Effectiveness: {baseline_report['average_effectiveness']:.2f}")
        print(f"Susceptibility Reduction: {baseline_report['susceptibility_reduction']:.2f}")
        print(f"Most Used Intervention: {baseline_report['most_used_intervention']}")
        
        # Log results
        logging.info("Simulation completed successfully")
        
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        raise

# -------------------------------
# Execute the Comparison
# -------------------------------

if __name__ == "__main__":
    main_comparison()
