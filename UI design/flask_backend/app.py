"""
NeuroFoldNet Flask API
Production-ready backend for tau protein classification
Integrates with Next.js frontend (NeuroFoldNet)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import csv
import io
import logging
from typing import Dict, List, Any
from datetime import datetime
import traceback

# Scikit-learn imports (needed for pickle to load the model)
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for Next.js frontend - allow all endpoints
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# Global variables
MODEL_PACKAGE = None
FEATURE_SCHEMA = None


# ============================================================================
# NeuroFoldNet Model Class Definition
# This MUST be defined before loading the pickle file!
# ============================================================================

class NeuroFoldNet(BaseEstimator, ClassifierMixin):
    """
    NeuroFoldNet: Advanced Triple-Layer Stacking Ensemble Classifier
    
    This model implements proper stacking with out-of-fold predictions to prevent data leakage.
    
    Architecture:
    - Layer 1: 5 diverse base learners (XGBoost x2, GB, SVM x2)
    - Layer 2: 2 meta-learners (XGBoost, Gradient Boosting)
    - Layer 3: 1 final combiner (XGBoost)
    """
    
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Layer 1: Base models (will store multiple copies for each fold)
        self.layer1_models = []
        # Layer 2: Meta-learners (will store multiple copies for each fold)
        self.layer2_models = []
        # Layer 3: Final meta-model
        self.meta_model = None
        
        # Final models trained on full dataset
        self.layer1_final = []
        self.layer2_final = []
        
    def _get_base_models(self):
        """Get fresh base model instances"""
        return [
            ('xgb_deep', xgb.XGBClassifier(
                n_estimators=250, max_depth=8, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.3, reg_lambda=2.0,
                random_state=self.random_state, eval_metric='mlogloss'
            )),
            ('xgb_wide', xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.05, reg_lambda=0.5,
                random_state=self.random_state, eval_metric='mlogloss'
            )),
            ('gb_tuned', GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.85, min_samples_split=4,
                random_state=self.random_state
            )),
            ('svm_rbf', SVC(
                C=15.0, kernel='rbf', gamma='auto',
                probability=True, random_state=self.random_state
            )),
            ('svm_poly', SVC(
                C=8.0, kernel='poly', degree=3,
                probability=True, random_state=self.random_state
            ))
        ]
    
    def _get_layer2_models(self):
        """Get fresh Layer 2 model instances"""
        return [
            ('xgb_meta', xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=self.random_state, eval_metric='mlogloss'
            )),
            ('gb_meta', GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=self.random_state
            ))
        ]
    
    def fit(self, X, y):
        """Train the model using proper out-of-fold stacking."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        n_samples = X_scaled.shape[0]
        n_classes = len(np.unique(y))
        
        # Layer 1: Out-of-Fold Predictions
        base_models = self._get_base_models()
        n_base_models = len(base_models)
        oof_layer1 = np.zeros((n_samples, n_base_models * n_classes))
        
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        self.layer1_models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
            X_train_fold = X_scaled[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_scaled[val_idx]
            
            fold_models = []
            
            for model_idx, (name, model) in enumerate(base_models):
                model_clone = clone(model)
                model_clone.fit(X_train_fold, y_train_fold)
                
                val_proba = model_clone.predict_proba(X_val_fold)
                oof_layer1[val_idx, model_idx*n_classes:(model_idx+1)*n_classes] = val_proba
                
                fold_models.append((name, model_clone))
            
            self.layer1_models.append(fold_models)
        
        # Train final Layer 1 models on full dataset
        self.layer1_final = []
        for name, model in base_models:
            model_clone = clone(model)
            model_clone.fit(X_scaled, y)
            self.layer1_final.append((name, model_clone))
        
        # Layer 2: Out-of-Fold Predictions
        layer2_models = self._get_layer2_models()
        n_layer2_models = len(layer2_models)
        oof_layer2 = np.zeros((n_samples, n_layer2_models * n_classes))
        
        self.layer2_models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(oof_layer1, y)):
            X_train_fold = oof_layer1[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = oof_layer1[val_idx]
            
            fold_models = []
            
            for model_idx, (name, model) in enumerate(layer2_models):
                model_clone = clone(model)
                model_clone.fit(X_train_fold, y_train_fold)
                
                val_proba = model_clone.predict_proba(X_val_fold)
                oof_layer2[val_idx, model_idx*n_classes:(model_idx+1)*n_classes] = val_proba
                
                fold_models.append((name, model_clone))
            
            self.layer2_models.append(fold_models)
        
        # Train final Layer 2 models on full dataset
        self.layer2_final = []
        for name, model in layer2_models:
            model_clone = clone(model)
            model_clone.fit(oof_layer1, y)
            self.layer2_final.append((name, model_clone))
        
        # Layer 3: Final Meta-Learner
        meta_features = np.hstack([oof_layer1, oof_layer2])
        
        self.meta_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            reg_alpha=0.2, reg_lambda=1.5,
            random_state=self.random_state, eval_metric='mlogloss'
        )
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        
        # Layer 1 predictions
        layer1_preds = []
        for name, model in self.layer1_final:
            layer1_preds.append(model.predict_proba(X_scaled))
        layer1_features = np.hstack(layer1_preds)
        
        # Layer 2 predictions
        layer2_preds = []
        for name, model in self.layer2_final:
            layer2_preds.append(model.predict_proba(layer1_features))
        layer2_features = np.hstack(layer2_preds)
        
        # Combine and predict with Layer 3
        meta_features = np.hstack([layer1_features, layer2_features])
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        X_scaled = self.scaler.transform(X)
        
        # Layer 1 predictions
        layer1_preds = []
        for name, model in self.layer1_final:
            layer1_preds.append(model.predict_proba(X_scaled))
        layer1_features = np.hstack(layer1_preds)
        
        # Layer 2 predictions
        layer2_preds = []
        for name, model in self.layer2_final:
            layer2_preds.append(model.predict_proba(layer1_features))
        layer2_features = np.hstack(layer2_preds)
        
        # Combine and predict with Layer 3
        meta_features = np.hstack([layer1_features, layer2_features])
        return self.meta_model.predict_proba(meta_features)


def load_model():
    """Load the trained NeuroFoldNet model"""
    global MODEL_PACKAGE
    
    try:
        logger.info("Loading NeuroFoldNet model...")
        with open('neurofoldnet_model.pkl', 'rb') as f:
            MODEL_PACKAGE = pickle.load(f)
        
        logger.info("Model loaded successfully!")
        logger.info(f"  Test Accuracy: {MODEL_PACKAGE['test_accuracy']:.4f}")
        logger.info(f"  Test F1-Score: {MODEL_PACKAGE['test_f1']:.4f}")
        logger.info(f"  Features: {MODEL_PACKAGE['feature_names']}")
        
        return True
    except FileNotFoundError:
        logger.error("Model file 'neurofoldnet_model.pkl' not found!")
        logger.error("Please ensure the model file is in the same directory as app.py")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def initialize_feature_schema():
    """Initialize feature schema matching the frontend"""
    global FEATURE_SCHEMA
    
    # This matches your frontend's features_schema.json
    FEATURE_SCHEMA = {
        "model_features": ["height", "area", "diameter", "proteo_resist_05", 
                          "proteo_resist_1", "seed_density", "basal_trans", "induced_trans"],
        "frontend_mapping": {
            # Figure 1 - AFM & CD (Morphology)
            "height": ["fig1_afm_height_mean"],
            "area": ["fig1_afm_area_mean"],
            "diameter": ["fig1_afm_diameter_mean"],
            
            # Figure 3 - Proteolytic Resistance
            "proteo_resist_05": ["fig3_pronase_0p5ug_1min"],
            "proteo_resist_1": ["fig3_pronase_1p0ug_1min"],
            
            # Figure 4 - Antibody (Seeding)
            "seed_density": ["fig4_mean_signal"],
            
            # Figure 6 - Electrophysiology
            "basal_trans": ["fig6_io_response_100"],
            "induced_trans": ["fig6_ltp_early_mean"]
        }
    }
    
    logger.info("Feature schema initialized")


def map_frontend_to_model(frontend_data: Dict[str, float]) -> np.ndarray:
    """
    Map frontend feature names to model's expected 8 core features.
    Uses intelligent aggregation when multiple frontend features map to one model feature.
    """
    
    try:
        model_features = []
        
        for model_feat in FEATURE_SCHEMA["model_features"]:
            frontend_keys = FEATURE_SCHEMA["frontend_mapping"][model_feat]
            
            # Collect values for this model feature
            values = []
            for key in frontend_keys:
                if key in frontend_data:
                    values.append(frontend_data[key])
            
            # Use mean if multiple values, otherwise use the single value
            if values:
                model_features.append(np.mean(values))
            else:
                # Use default value if missing (from training data statistics)
                defaults = {
                    "height": 1.1549,
                    "area": 70.5647,
                    "diameter": 7.8716,
                    "proteo_resist_05": 53.9857,
                    "proteo_resist_1": 28.6163,
                    "seed_density": 0.1583,
                    "basal_trans": 0.3435,
                    "induced_trans": 101.5858
                }
                model_features.append(defaults[model_feat])
                logger.warning(f"Missing data for {model_feat}, using default: {defaults[model_feat]}")
        
        return np.array(model_features).reshape(1, -1)
    
    except Exception as e:
        logger.error(f"Error mapping features: {str(e)}")
        raise ValueError(f"Invalid feature data: {str(e)}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if MODEL_PACKAGE is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded',
            'timestamp': datetime.now().isoformat()
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'model': 'NeuroFoldNet',
        'version': '2.0',
        'timestamp': datetime.now().isoformat(),
        'features_count': len(MODEL_PACKAGE['feature_names']),
        'classes': MODEL_PACKAGE['label_encoder'].classes_.tolist()
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information and performance metrics"""
    if MODEL_PACKAGE is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    cv_summary = MODEL_PACKAGE.get('cv_summary', [])
    
    # Find NeuroFoldNet results
    neurofoldnet_results = next(
        (item for item in cv_summary if item['Model'] == 'NeuroFoldNet'),
        {}
    )
    
    return jsonify({
        'model_name': 'NeuroFoldNet',
        'version': '2.0',
        'description': 'Triple-layer stacking ensemble for tau protein classification',
        'architecture': {
            'layer1': {
                'description': '5 diverse base learners',
                'models': ['XGBoost (deep)', 'XGBoost (wide)', 'Gradient Boosting', 'SVM (RBF)', 'SVM (Poly)']
            },
            'layer2': {
                'description': '2 meta-learners',
                'models': ['XGBoost', 'Gradient Boosting']
            },
            'layer3': {
                'description': '1 final combiner',
                'models': ['XGBoost']
            },
            'total_models': 8,
            'stacking_method': 'Out-of-fold (no data leakage)',
            'cv_folds': 5
        },
        'performance': {
            'test_accuracy': round(MODEL_PACKAGE['test_accuracy'], 4),
            'test_f1_score': round(MODEL_PACKAGE['test_f1'], 4),
            'test_log_loss': round(MODEL_PACKAGE.get('test_logloss', 0.0), 4),
            'cv_accuracy_mean': round(neurofoldnet_results.get('Accuracy_Mean', 0), 4),
            'cv_accuracy_std': round(neurofoldnet_results.get('Accuracy_Std', 0), 4),
            'cv_f1_mean': round(neurofoldnet_results.get('F1_Mean', 0), 4),
            'cv_f1_std': round(neurofoldnet_results.get('F1_Std', 0), 4)
        },
        'classes': MODEL_PACKAGE['label_encoder'].classes_.tolist(),
        'features': MODEL_PACKAGE['feature_names'],
        'training_date': '2026-01-22',
        'methodology': 'Proper out-of-fold stacking with stratified K-fold cross-validation'
    })


@app.route('/api/predict', methods=['POST'])
def predict_single():
    """
    Single sample prediction endpoint
    
    Expected JSON format:
    {
        "features": {
            "fig1_afm_height_mean": 1.1549,
            "fig1_afm_area_mean": 70.5647,
            ...
        }
    }
    """
    if MODEL_PACKAGE is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid request format. Expected {"features": {...}}'}), 400
        
        features = data['features']
        
        # Map frontend features to model features
        X = map_frontend_to_model(features)
        
        # Get predictions
        model = MODEL_PACKAGE['model']
        label_encoder = MODEL_PACKAGE['label_encoder']
        
        prediction_class = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Build response
        predicted_disease = label_encoder.classes_[prediction_class]
        
        # Create probability breakdown
        prob_breakdown = {}
        for i, disease in enumerate(label_encoder.classes_):
            prob_breakdown[disease] = {
                'probability': round(float(probabilities[i]), 4),
                'percentage': round(float(probabilities[i] * 100), 2)
            }
        
        # Determine confidence level
        max_prob = max(probabilities)
        if max_prob >= 0.90:
            confidence = 'high'
        elif max_prob >= 0.70:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        response = {
            'prediction': predicted_disease,
            'confidence': confidence,
            'probabilities': prob_breakdown,
            'probability_vector': probabilities.tolist(),
            'metadata': {
                'model': 'NeuroFoldNet v2.0',
                'timestamp': datetime.now().isoformat(),
                'features_received': len(features),
                'features_used': len(MODEL_PACKAGE['feature_names'])
            }
        }
        
        logger.info(f"Prediction: {predicted_disease} (confidence: {max_prob:.2%})")
        
        return jsonify(response)
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error during prediction'}), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint for CSV files
    
    Expected: CSV file with feature columns matching frontend schema
    Returns: JSON array of predictions
    """
    if MODEL_PACKAGE is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be CSV format'}), 400
        
        # Read CSV without pandas (avoid blocked native DLLs)
        try:
            content = file.stream.read().decode('utf-8-sig')
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            logger.info(f"Loaded CSV with {len(rows)} rows and {len(reader.fieldnames or [])} columns")
        except Exception as e:
            return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 400
        
        # Process each row
        predictions = []
        model = MODEL_PACKAGE['model']
        label_encoder = MODEL_PACKAGE['label_encoder']
        
        def parse_float(value):
            if value is None:
                return None
            if isinstance(value, str):
                value = value.strip()
                if value == "" or value.lower() in {"nan", "na", "null"}:
                    return None
            try:
                return float(value)
            except Exception:
                return None

        for idx, row in enumerate(rows):
            try:
                # Convert row values to floats where possible
                features = {}
                for key, value in row.items():
                    parsed = parse_float(value)
                    if parsed is not None:
                        features[key] = parsed
                
                # Map to model features
                X = map_frontend_to_model(features)
                
                # Predict
                pred_class = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                
                predicted_disease = label_encoder.classes_[pred_class]
                
                # Build result
                prob_breakdown = {}
                for i, disease in enumerate(label_encoder.classes_):
                    prob_breakdown[disease] = round(float(probabilities[i]), 4)
                
                predictions.append({
                    'row_index': int(idx),
                    'prediction': predicted_disease,
                    'probabilities': prob_breakdown,
                    'max_probability': round(float(max(probabilities)), 4)
                })
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                predictions.append({
                    'row_index': int(idx),
                    'error': str(e)
                })
        
        # Count predictions
        pred_counts = {}
        for p in predictions:
            if 'prediction' in p:
                disease = p['prediction']
                pred_counts[disease] = pred_counts.get(disease, 0) + 1
        
        response = {
            'total_samples': len(rows),
            'successful_predictions': len([p for p in predictions if 'prediction' in p]),
            'failed_predictions': len([p for p in predictions if 'error' in p]),
            'prediction_summary': pred_counts,
            'predictions': predictions,
            'metadata': {
                'model': 'NeuroFoldNet v2.0',
                'timestamp': datetime.now().isoformat(),
                'filename': file.filename
            }
        }
        
        logger.info(f"Batch prediction complete: {len(predictions)} samples processed")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error during batch prediction'}), 500


@app.route('/api/feature-schema', methods=['GET'])
def get_feature_schema():
    """Return the feature schema for frontend validation"""
    if FEATURE_SCHEMA is None:
        return jsonify({'error': 'Feature schema not initialized'}), 503
    
    return jsonify({
        'model_features': FEATURE_SCHEMA['model_features'],
        'frontend_mapping': FEATURE_SCHEMA['frontend_mapping'],
        'total_frontend_features': sum(len(v) for v in FEATURE_SCHEMA['frontend_mapping'].values()),
        'total_model_features': len(FEATURE_SCHEMA['model_features'])
    })


@app.route('/api/validate-features', methods=['POST'])
def validate_features():
    """Validate feature data before prediction"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        features = data['features']
        
        # Check for missing values
        missing_features = []
        for model_feat, frontend_keys in FEATURE_SCHEMA['frontend_mapping'].items():
            has_any = any(key in features for key in frontend_keys)
            if not has_any:
                missing_features.append(model_feat)
        
        # Check for invalid values
        invalid_features = []
        for key, value in features.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                invalid_features.append(key)
        
        valid = len(missing_features) == 0 and len(invalid_features) == 0
        
        return jsonify({
            'valid': valid,
            'missing_features': missing_features,
            'invalid_features': invalid_features,
            'total_features_provided': len(features),
            'warnings': [f"Missing data for: {', '.join(missing_features)}" if missing_features else None,
                        f"Invalid values in: {', '.join(invalid_features)}" if invalid_features else None]
        })
    
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/performance-metrics', methods=['GET'])
def performance_metrics():
    """Get detailed performance metrics from cross-validation"""
    if MODEL_PACKAGE is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    cv_summary = MODEL_PACKAGE.get('cv_summary', [])
    
    # Format for frontend display
    formatted_results = []
    for model_result in cv_summary:
        formatted_results.append({
            'model_name': model_result['Model'],
            'accuracy': {
                'mean': round(model_result['Accuracy_Mean'], 4),
                'std': round(model_result['Accuracy_Std'], 4)
            },
            'f1_score': {
                'mean': round(model_result['F1_Mean'], 4),
                'std': round(model_result['F1_Std'], 4)
            },
            'log_loss': {
                'mean': round(model_result.get('LogLoss_Mean', 0), 4),
                'std': round(model_result.get('LogLoss_Std', 0), 4)
            }
        })
    
    # Sort by accuracy
    formatted_results.sort(key=lambda x: x['accuracy']['mean'], reverse=True)
    
    return jsonify({
        'cross_validation': {
            'method': 'Stratified Group K-Fold',
            'n_folds': 5,
            'results': formatted_results
        },
        'test_set': {
            'accuracy': round(MODEL_PACKAGE['test_accuracy'], 4),
            'f1_score': round(MODEL_PACKAGE['test_f1'], 4),
            'log_loss': round(MODEL_PACKAGE.get('test_logloss', 0.0), 4)
        },
        'best_model': formatted_results[0]['model_name'] if formatted_results else None
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("="*80)
    print(" NeuroFoldNet Flask API Server")
    print("="*80)
    print()
    
    # Initialize
    initialize_feature_schema()
    
    # Load model
    if not load_model():
        print("Failed to load model. Exiting...")
        exit(1)
    
    print()
    print("Server initialization complete!")
    print()
    print("API Endpoints:")
    print("   GET  /health                    - Health check")
    print("   GET  /api/model-info            - Model information")
    print("   POST /api/predict               - Single prediction")
    print("   POST /api/predict-batch         - Batch predictions (CSV)")
    print("   GET  /api/feature-schema        - Feature schema")
    print("   POST /api/validate-features     - Validate feature data")
    print("   GET  /api/performance-metrics   - Performance metrics")
    print()
    print("Starting server on http://localhost:5000")
    print("   CORS enabled for: http://localhost:3000")
    print()
    print("="*80)
    print()
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5000)
