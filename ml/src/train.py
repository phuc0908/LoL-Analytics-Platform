"""
Train Win Prediction Model for LoL Esports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import json
from datetime import datetime

from data_processor import LoLDataProcessor


class WinPredictionModel:
    """XGBoost model for predicting match outcomes"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train the model"""
        print("🚀 Starting model training...")
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📊 Train size: {len(X_train):,}, Test size: {len(X_test):,}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost (without early stopping for CV compatibility)
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
        )
        
        self.model.fit(X_train_scaled, y_train, verbose=False)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"\n✅ Model trained successfully!")
        print(f"📈 Test Accuracy: {accuracy*100:.2f}%")
        print(f"📈 ROC AUC: {auc:.4f}")
        
        # Cross-validation with a fresh model
        cv_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        cv_scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"📈 CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        
        # Store metrics
        self.metrics = {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': len(self.feature_names),
            'trained_at': datetime.now().isoformat(),
        }
        
        # Feature importance
        print("\n📊 Top 10 Feature Importances:")
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(importance.head(10).to_string(index=False))
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Red Wins', 'Blue Wins']))
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> dict:
        """Predict match outcome"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)[0]
        
        return {
            'blue_win_probability': float(prob[1]),
            'red_win_probability': float(prob[0]),
            'predicted_winner': 'Blue' if prob[1] > 0.5 else 'Red',
            'confidence': float(max(prob))
        }
    
    def save(self, model_dir: str = "../models"):
        """Save model and scaler"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_dir / "win_predictor.joblib")
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'metrics': self.metrics,
        }
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to {model_dir}")
    
    def load(self, model_dir: str = "../models"):
        """Load model and scaler"""
        model_dir = Path(model_dir)
        
        self.model = joblib.load(model_dir / "win_predictor.joblib")
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.metrics = metadata['metrics']
        
        print(f"✅ Model loaded from {model_dir}")
        print(f"📈 Model accuracy: {self.metrics['accuracy']*100:.2f}%")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🎮 LoL Win Prediction Model Training")
    print("=" * 60)
    
    # Load and process data
    data_path = Path(__file__).parent.parent.parent / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
    
    processor = LoLDataProcessor(str(data_path))
    processor.load_data()
    
    # Prepare ML data (with early game stats for better accuracy)
    X, y = processor.prepare_ml_data(use_early_game=True)
    
    # Train model
    model = WinPredictionModel()
    metrics = model.train(X, y)
    
    # Save model
    model.save()
    
    # Save stats for API
    print("\n📊 Generating statistics...")
    stats_dir = Path(__file__).parent.parent / "data"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Champion stats
    champ_stats = processor.get_champion_stats()
    champ_stats.to_csv(stats_dir / "champion_stats.csv", index=False)
    print(f"✅ Saved champion stats ({len(champ_stats)} champions)")
    
    # Team stats
    team_stats = processor.get_team_stats()
    team_stats.to_csv(stats_dir / "team_stats.csv", index=False)
    print(f"✅ Saved team stats ({len(team_stats)} teams)")
    
    # Player stats
    player_stats = processor.get_player_stats()
    player_stats.to_csv(stats_dir / "player_stats.csv", index=False)
    print(f"✅ Saved player stats ({len(player_stats)} players)")
    
    # League stats
    league_stats = processor.get_league_stats()
    league_stats.to_csv(stats_dir / "league_stats.csv", index=False)
    print(f"✅ Saved league stats ({len(league_stats)} leagues)")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    
    return model, processor


if __name__ == "__main__":
    main()

