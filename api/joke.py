import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ExercisePredictor:
    def __init__(self):
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        self.data = sns.load_dataset('exercise')
        
    def preprocess_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Define a good heart rate range
        self.data['good_hr'] = (self.data['pulse'] >= 100) & (self.data['pulse'] <= 150)
        
        # Convert categorical variables into dummy variables
        self.data = pd.get_dummies(self.data, columns=['diet', 'kind'])
        
    def train_model(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() and preprocess_data() first.")
        
        X = self.data.drop(['id', 'time', 'good_hr'], axis=1)
        y = self.data['good_hr']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        
    def predict_heart_rate(self, data):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Preprocess input data
        data = pd.get_dummies(data, columns=['diet', 'kind'])
        
        # Make predictions
        predictions = self.model.predict(data)
        return predictions

# Example usage
def main():
    # Create ExercisePredictor object
    predictor = ExercisePredictor()
    
    # Load and preprocess data
    predictor.load_data()
    predictor.preprocess_data()
    
    # Train the model
    predictor.train_model()
    
    # Evaluate the model
    predictor.evaluate_model()
    
    # Define new data for prediction
    new_data = pd.DataFrame({
        'diet_no fat': [0],  # Example diet type (0 for 'low fat', 1 for 'no fat')
        'pulse': [140],  # Example heart rate
        'diet_time': [1],  # Example time of day (1 for morning, 2 for afternoon, 3 for evening)
        'diet_low fat': [1],  # Example diet type (0 for 'low fat', 1 for 'no fat')
        'kind_rest': [0],  # Example exercise type (0 for 'rest', 1 for 'walking', 0 for 'running')
        'kind_running': [0]
    })
    
    # Make predictions
    predictions = predictor.predict_heart_rate(new_data)
    print("Predicted heart rate status:", "good" if predictions[0] == 1 else "not good")

if __name__ == "__main__":
    main()
