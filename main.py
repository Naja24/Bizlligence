import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# Append the project root directory (or the src directory if you prefer)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import preprocessing pipeline and supervised learning modules
from preprocessing import PreprocessingPipeline
from model.supervised import SupervisedLearning

# Define dataset path and extract dataset name
dataset_path = "Titanic-Dataset.csv"
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

# Load dataset
data = pd.read_csv(dataset_path)
df_sample = pd.DataFrame(data)

# Initialize and run the preprocessing pipeline
preproc_pipeline = PreprocessingPipeline(
    impute_type='simple',
    strategy='mean',
    n_neighbors=5,
    task_type='auto',  # Set to 'auto' to detect task type automatically
    target_col='Survived',
    numeric_features=None,          # Auto-detect numeric features
    categorical_features=None,      # Auto-detect categorical features
    scale_method='minmax',
    normalize=True,                 # Apply normalization after scaling
    use_poly=False,
    poly_degree=2
)

X_processed, y_processed = preproc_pipeline.process(df_sample)
print("Processed features:")
print(X_processed.head())
print("\nProcessed target (encoded):")
print(y_processed.head())

# Initialize supervised learning module with auto task detection
supervised_model = SupervisedLearning(
    task_type='auto',  # Set to 'auto' to detect task type automatically
    models_to_train=None,  # This will train all applicable models for the detected type
    random_state=42,
    n_jobs=-1,
    verbose=True
)

# Split the data into training and test sets
supervised_model.split_data(X_processed, y_processed, test_size=0.2, stratify=True)

# Output detected task type
print(f"\nDetected Task Type: {supervised_model.task_type}")

if __name__ == '__main__':
    # Import multiprocessing and add freeze_support
    import multiprocessing
    multiprocessing.freeze_support()

    # Train the models
    print("\nTraining all suitable models...")
    supervised_model.train()

    # Print the best model selected
    print(f"\nBest model selected: {supervised_model.best_model_name}")

    # Perform hyperparameter tuning on the best model
    print("\nPerforming hyperparameter tuning on the best model...")
    supervised_model.tune_hyperparameters(
        model_name=supervised_model.best_model_name,
        cv=5,
        n_jobs=-1
    )

    # Print best model hyperparameters
    print(f"\nBest model hyperparameters ({supervised_model.best_model_name}):")
    print(supervised_model.best_model.get_params())

    # Evaluate the models on test data
    print("\nEvaluating tuned models on test data...")
    evaluation_results = supervised_model.evaluate()
    print("\nEvaluation Results:")
    for model_name, metrics in evaluation_results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            elif metric not in ['confusion_matrix', 'classification_report']:
                print(f"  {metric}: {value}")

    # Handle classification-specific metrics if applicable
    if supervised_model.task_type == 'classification':
        best_model_metrics = evaluation_results.get(supervised_model.best_model_name, {})
        if 'confusion_matrix' in best_model_metrics:
            print(f"\nConfusion Matrix for {supervised_model.best_model_name}:")
            print(best_model_metrics['confusion_matrix'])

        if 'classification_report' in best_model_metrics:
            print(f"\nClassification Report for {supervised_model.best_model_name}:")
            print(best_model_metrics['classification_report'])

    # Make predictions with the best model
    predictions = supervised_model.predict(supervised_model.X_test)
    print("\nSample Predictions (first 10):")
    print(predictions[:10])

    # If classification, try to get probability predictions
    if supervised_model.task_type == 'classification':
        try:
            prediction_probs = supervised_model.predict_proba(supervised_model.X_test)
            print("\nSample Prediction Probabilities (first 5):")
            for i in range(min(5, len(prediction_probs))):
                prob_str = ", ".join([f"Class {j}: {p:.4f}" for j, p in enumerate(prediction_probs[i])])
                print(f"Sample {i+1}: {prob_str}")
        except Exception as e:
            print(f"\nCould not get prediction probabilities: {str(e)}")

    # Plot feature importance if available
    try:
        print("\nPlotting feature importance...")
        fig = supervised_model.plot_feature_importance(top_n=10)
        plt_filename = f"{dataset_name}_{supervised_model.task_type}_feature_importance.png"
        fig.savefig(plt_filename)
        print(f"Feature importance plot saved as {plt_filename}")
    except Exception as e:
        print(f"Could not plot feature importance: {str(e)}")

    # Save the best model
    model_filename = f"{dataset_name}_{supervised_model.task_type}_best_model.joblib"
    supervised_model.save_model(model_filename)
    print(f"\nBest model saved as {model_filename}")

    # Show model summary
    print("\nModel Summary:")
    summary = supervised_model.get_summary()
    for key, value in summary.items():
        if key != 'evaluation_results' and key != 'top_features':
            print(f"{key}: {value}")

    if 'top_features' in summary:
        print("\nTop Features:")
        for i, feature in enumerate(summary['top_features']):
            print(f"{i+1}. {feature['feature']}: {feature['importance']:.4f}")
