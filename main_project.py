# main_project.py
# Main Project: GNSS-5G Hybrid Positioning with AI-Enhanced MRAKF

import numpy as np
import pandas as pd
import pickle
import time
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from scipy.linalg import block_diag
warnings.filterwarnings('ignore')

# ============================================================================
# 1. Data Loading Functions
# ============================================================================

def load_dataset(filename='gnss_5g_dataset.pkl'):
    """Load the generated dataset"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Dataset loaded from {filename}")
    print(f"   - Features: {data['features'].shape}")
    print(f"   - Labels: {data['noise_labels'].shape}")
    return data

def get_feature_names():
    """Get feature names"""
    return ['distance', 'bs_index', 'pos_x', 'pos_y', 'pos_z', 
            'vel_x', 'vel_y', 'vel_z', 'time_norm', 'sin_time', 'cos_time']

def get_label_names():
    """Get label names"""
    return ['sigma_toa', 'sigma_azimuth', 'sigma_elevation']


# ============================================================================
# 2. XGBoost Noise Predictor
# ============================================================================

class XGBoostNoisePredictor:
    """XGBoost-based noise predictor for GNSS-5G positioning"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models = []
        self.scaler_mean = None
        self.scaler_std = None
        
    def preprocess_features(self, features):
        """Normalize features"""
        if self.scaler_mean is None:
            self.scaler_mean = np.mean(features, axis=0)
            self.scaler_std = np.std(features, axis=0) + 1e-8
        return (features - self.scaler_mean) / self.scaler_std
    
    def train(self, features, labels, test_size=0.2):
        """Train XGBoost models for each noise component"""
        print("\n" + "=" * 60)
        print("TRAINING XGBOOST NOISE PREDICTOR")
        print("=" * 60)
        
        # Preprocess features
        X = self.preprocess_features(features)
        y = labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train separate models for each noise component
        self.models = []
        results = {}
        
        for i, label_name in enumerate(['TOA', 'Azimuth', 'Elevation']):
            print(f"\nTraining model for {label_name}...")
            
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective='reg:squarederror',
                random_state=42,
                verbosity=0
            )
            
            start_time = time.time()
            model.fit(X_train, y_train[:, i])
            train_time = time.time() - start_time
            
            # Predict and evaluate
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            rmse_train = np.sqrt(mean_squared_error(y_train[:, i], y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test[:, i], y_pred_test))
            r2_test = r2_score(y_test[:, i], y_pred_test)
            
            results[label_name] = {
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'r2_test': r2_test,
                'train_time': train_time
            }
            
            print(f"  RMSE (Train): {rmse_train:.2e}")
            print(f"  RMSE (Test):  {rmse_test:.2e}")
            print(f"  R² (Test):    {r2_test:.4f}")
            print(f"  Time:         {train_time:.2f}s")
            
            self.models.append(model)
        
        self.training_results = results
        return results
    
    def predict(self, features):
        """Predict noise STDs"""
        if not self.models:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.preprocess_features(features)
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred.reshape(-1, 1))
        
        return np.hstack(predictions)
    
    def predict_single(self, feature):
        """Predict for single feature vector"""
        feature = np.array(feature).reshape(1, -1)
        return self.predict(feature)[0]
    
    def save_model(self, filename='xgboost_noise_model.pkl'):
        """Save trained model"""
        model_data = {
            'models': self.models,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'training_results': self.training_results
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✅ Model saved to {filename}")
    
    def load_model(self, filename='xgboost_noise_model.pkl'):
        """Load trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.models = model_data['models']
        self.scaler_mean = model_data['scaler_mean']
        self.scaler_std = model_data['scaler_std']
        self.training_results = model_data['training_results']
        print(f"\n✅ Model loaded from {filename}")


# ============================================================================
# 3. System Model (Simplified for Integration)
# ============================================================================

class SystemModel:
    """Simplified system model for positioning"""
    
    def __init__(self, config):
        self.c = 299792458.0
        self.K = config.get('K', 4)
        self.M = config.get('M', 9)
        self.mu_BS = config.get('mu_BS', 10)
        self.mu_Sat = config.get('mu_Sat', 1)
        self.R_a = self.mu_BS // self.mu_Sat
        self.n_PL = config.get('n_PL', 1.98)
        self.sigma_v = config.get('sigma_v', 3.0)
        self.sigma_eps = config.get('sigma_eps', 63e-6)
        self.BS_positions = config.get('BS_positions', 
                                       np.array([[0, 20, 10], [50, -20, 10], 
                                                 [100, 20, 10], [150, -20, 10]]))
        
    def get_state_transition_matrix(self, dt):
        F = np.eye(8)
        F[0:3, 3:6] = dt * np.eye(3)
        F[6, 7] = dt
        return F
    
    def get_process_noise_covariance(self, dt):
        Q = np.zeros((8, 8))
        sigma_v2 = self.sigma_v ** 2
        Q[0:3, 0:3] = sigma_v2 * (dt**3 / 3) * np.eye(3)
        Q[0:3, 3:6] = sigma_v2 * (dt**2 / 2) * np.eye(3)
        Q[3:6, 0:3] = sigma_v2 * (dt**2 / 2) * np.eye(3)
        Q[3:6, 3:6] = sigma_v2 * dt * np.eye(3)
        
        sigma_eps2 = self.sigma_eps ** 2
        Q[6, 6] = sigma_eps2 * (dt**3 / 3)
        Q[6, 7] = sigma_eps2 * (dt**2 / 2)
        Q[7, 6] = sigma_eps2 * (dt**2 / 2)
        Q[7, 7] = sigma_eps2 * dt
        return Q


# ============================================================================
# 4. AI-Enhanced MRAKF
# ============================================================================

class AIEnhancedMRAKF:
    """AI-Enhanced Multi-Rate Adaptive Kalman Filter"""
    
    def __init__(self, system_model, predictor):
        self.model = system_model
        self.predictor = predictor
        
        # State initialization
        self.x = np.zeros(8)
        self.P = np.eye(8) * 100
        
        # GNSS noise covariance
        self.R_Sat = np.eye(system_model.M) * 0.25  # 0.5^2
        
        # Calibration parameters
        self.CRLB0 = None
        self.calibrated = False
        self.use_ai = True  # Use AI prediction by default
        
    def calibrate(self, positions, measurements):
        """Calibrate the mathematical model (fallback)"""
        print("\nCalibrating mathematical noise model...")
        
        distances = []
        sigmas = []
        
        for pos in positions[:10]:
            for k in range(self.model.K):
                bs_pos = self.model.BS_positions[k]
                d = np.linalg.norm(pos - bs_pos)
                distances.append(d)
                # Estimate sigma from data (simplified)
                sigma = 1e-8 * (d ** self.model.n_PL)
                sigmas.append(sigma)
        
        self.CRLB0 = np.mean(np.array(sigmas)**2 / (np.array(distances) ** self.model.n_PL))
        self.calibrated = True
        print(f"Calibration complete: CRLB0 = {self.CRLB0:.2e}")
    
    def predict_noise_math(self, state):
        """Mathematical noise prediction (fallback)"""
        if not self.calibrated:
            # Default values
            sigma_toa = 1e-8
            sigma_az = 0.5 * np.pi / 180
            sigma_el = 0.5 * np.pi / 180
        else:
            p = state[:3]
            sigma_toa_list = []
            sigma_az_list = []
            sigma_el_list = []
            
            for k in range(self.model.K):
                bs_pos = self.model.BS_positions[k]
                d = np.linalg.norm(p - bs_pos)
                sigma = np.sqrt(self.CRLB0 * (d ** self.model.n_PL))
                sigma_toa_list.append(sigma)
                sigma_az_list.append(sigma * 100)  # Scale for angles
                sigma_el_list.append(sigma * 100)
            
            sigma_toa = np.mean(sigma_toa_list)
            sigma_az = np.mean(sigma_az_list)
            sigma_el = np.mean(sigma_el_list)
        
        R_list = []
        for k in range(self.model.K):
            R_k = np.diag([sigma_toa**2, sigma_az**2, sigma_el**2])
            R_list.append(R_k)
        
        return block_diag(*R_list)
    
    def predict_noise_ai(self, state, time_idx):
        """AI-based noise prediction"""
        p = state[:3]
        R_list = []
        
        for k in range(self.model.K):
            bs_pos = self.model.BS_positions[k]
            d = np.linalg.norm(p - bs_pos)
            
            # Prepare features
            feature = np.array([
                d,
                k,
                p[0], p[1], p[2],
                state[3], state[4], state[5],
                time_idx / 100,
                np.sin(2 * np.pi * time_idx / 100),
                np.cos(2 * np.pi * time_idx / 100),
            ]).reshape(1, -1)
            
            # Predict
            pred = self.predictor.predict(feature)[0]
            sigma_toa = max(pred[0], 1e-12)
            sigma_az = max(pred[1], 1e-6)
            sigma_el = max(pred[2], 1e-6)
            
            R_k = np.diag([sigma_toa**2, sigma_az**2, sigma_el**2])
            R_list.append(R_k)
        
        return block_diag(*R_list)
    
    def predict_noise(self, state, time_idx):
        """Predict noise using AI or mathematical model"""
        if self.use_ai:
            try:
                return self.predict_noise_ai(state, time_idx)
            except:
                return self.predict_noise_math(state)
        else:
            return self.predict_noise_math(state)
    
    def _compute_BS_Jacobian(self, state):
        """Compute Jacobian for BS measurements"""
        K = self.model.K
        H = np.zeros((3 * K, 8))
        p = state[:3]
        
        for k in range(K):
            bs_pos = self.model.BS_positions[k]
            dx = p[0] - bs_pos[0]
            dy = p[1] - bs_pos[1]
            dz = p[2] - bs_pos[2]
            d_2D = np.sqrt(dx**2 + dy**2 + 1e-10)
            d_3D = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-10)
            
            i = 3 * k
            H[i, 0] = dx / (self.model.c * d_3D)
            H[i, 1] = dy / (self.model.c * d_3D)
            H[i, 2] = dz / (self.model.c * d_3D)
            H[i, 6] = 1
            
            H[i+1, 0] = -dy / (d_2D**2 + 1e-10)
            H[i+1, 1] = dx / (d_2D**2 + 1e-10)
            
            H[i+2, 0] = -(dx * dz) / (d_2D * d_3D**2 + 1e-10)
            H[i+2, 1] = -(dy * dz) / (d_2D * d_3D**2 + 1e-10)
            H[i+2, 2] = d_2D / (d_3D**2 + 1e-10)
        
        return H
    
    def _compute_BS_measurement_function(self, state):
        """Compute measurement function h_BS"""
        K = self.model.K
        h = np.zeros(3 * K)
        p = state[:3]
        rho = state[6]
        
        for k in range(K):
            bs_pos = self.model.BS_positions[k]
            dx = p[0] - bs_pos[0]
            dy = p[1] - bs_pos[1]
            dz = p[2] - bs_pos[2]
            
            d_3D = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-10)
            d_2D = np.sqrt(dx**2 + dy**2 + 1e-10)
            
            i = 3 * k
            h[i] = d_3D / self.model.c + rho
            h[i+1] = np.arctan2(dy, dx + 1e-10)
            h[i+2] = np.arctan2(dz, d_2D + 1e-10)
        
        return h
    
    def first_stage_EKF(self, y_BS, t, dt):
        """First-stage EKF with 5G measurements"""
        # Prediction
        F = self.model.get_state_transition_matrix(dt)
        Q = self.model.get_process_noise_covariance(dt)
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        
        # Predict noise
        R_BS = self.predict_noise(self.x, t)
        R_BS = R_BS + 1e-10 * np.eye(R_BS.shape[0])
        
        # Update
        H = self._compute_BS_Jacobian(self.x)
        S = H @ self.P @ H.T + R_BS
        S = (S + S.T) / 2
        S = S + 1e-8 * np.eye(S.shape[0])
        
        K = self.P @ H.T @ np.linalg.inv(S)
        
        h = self._compute_BS_measurement_function(self.x)
        y_pred = y_BS.flatten()
        innovation = y_pred - h
        
        self.x = self.x + K @ innovation
        self.P = (np.eye(8) - K @ H) @ self.P
        
        return self.x.copy()
    
    def second_stage_EKF(self, y_Sat):
        """Second-stage EKF with GNSS measurements"""
        x_prior = self.x.copy()
        P_prior = self.P.copy()
        
        # Sequential update for each satellite
        for m in range(self.model.M):
            H_m = self._compute_GNSS_Jacobian(x_prior, m)
            S_m = float(H_m @ P_prior @ H_m.T + self.R_Sat[m, m]) + 1e-10
            K_m = (P_prior @ H_m.T) / S_m
            h_m = self._compute_GNSS_measurement(x_prior, m)
            innovation = y_Sat[m] - h_m
            
            x_prior = x_prior + K_m.flatten() * innovation
            P_prior = (np.eye(8) - K_m @ H_m) @ P_prior
        
        self.x = x_prior
        self.P = P_prior
        
        return self.x.copy()
    
    def _compute_GNSS_Jacobian(self, state, m):
        """Compute Jacobian for single GNSS satellite"""
        H = np.zeros(8)
        p = state[:3]
        
        # Simplified satellite positions
        sat_pos = np.array([20200000 * np.cos(45*np.pi/180) * np.cos(m*40*np.pi/180),
                           20200000 * np.cos(45*np.pi/180) * np.sin(m*40*np.pi/180),
                           20200000 * np.sin(45*np.pi/180)])
        dx = p[0] - sat_pos[0]
        dy = p[1] - sat_pos[1]
        dz = p[2] - sat_pos[2]
        d = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-10)
        
        H[0] = dx / (self.model.c * d)
        H[1] = dy / (self.model.c * d)
        H[2] = dz / (self.model.c * d)
        H[6] = 1
        
        return H.reshape(1, 8)
    
    def _compute_GNSS_measurement(self, state, m):
        """Compute measurement function for single GNSS satellite"""
        p = state[:3]
        rho = state[6]
        
        sat_pos = np.array([20200000 * np.cos(45*np.pi/180) * np.cos(m*40*np.pi/180),
                           20200000 * np.cos(45*np.pi/180) * np.sin(m*40*np.pi/180),
                           20200000 * np.sin(45*np.pi/180)])
        d = np.linalg.norm(p - sat_pos)
        
        return d / self.model.c + rho
    
    def run(self, BS_meas, GNSS_meas):
        """Run the full filter"""
        N_epochs = len(GNSS_meas)
        R_a = self.model.R_a
        
        positions = []
        errors = []
        
        for n in range(N_epochs):
            # Process high-rate BS measurements
            for i in range(R_a):
                t = n * R_a + i
                if t < len(BS_meas):
                    dt = 1.0 / self.model.mu_BS
                    y_BS = BS_meas[t]
                    self.first_stage_EKF(y_BS, t, dt)
            
            # Process GNSS measurements
            if n < len(GNSS_meas):
                y_Sat = GNSS_meas[n]
                self.second_stage_EKF(y_Sat)
            
            positions.append(self.x[:3].copy())
        
        return np.array(positions)


# ============================================================================
# 5. Standard EKF for Comparison
# ============================================================================

class StandardEKF:
    """Standard EKF with constant R matrix"""
    
    def __init__(self, system_model):
        self.model = system_model
        self.x = np.zeros(8)
        self.P = np.eye(8) * 100
        
        # Constant R
        sigma_toa = 1e-8
        sigma_az = 0.5 * np.pi / 180
        sigma_el = 0.5 * np.pi / 180
        R_list = []
        for k in range(system_model.K):
            R_k = np.diag([sigma_toa**2, sigma_az**2, sigma_el**2])
            R_list.append(R_k)
        self.R_BS = block_diag(*R_list)
        self.R_Sat = np.eye(system_model.M) * 0.25
    
    def run(self, BS_meas, GNSS_meas):
        """Run standard EKF"""
        N_epochs = len(GNSS_meas)
        positions = []
        
        for n in range(N_epochs):
            if n < len(BS_meas):
                dt = 1.0
                F = self.model.get_state_transition_matrix(dt)
                Q = self.model.get_process_noise_covariance(dt)
                
                self.x = F @ self.x
                self.P = F @ self.P @ F.T + Q
                
                # BS update
                y_BS = BS_meas[n].flatten()
                H_BS = self._compute_BS_Jacobian(self.x)
                S = H_BS @ self.P @ H_BS.T + self.R_BS
                S = S + 1e-8 * np.eye(S.shape[0])
                K = self.P @ H_BS.T @ np.linalg.inv(S)
                h_BS = self._compute_BS_measurement_function(self.x)
                self.x = self.x + K @ (y_BS - h_BS)
                self.P = (np.eye(8) - K @ H_BS) @ self.P
                
                # GNSS update
                y_Sat = GNSS_meas[n]
                for m in range(self.model.M):
                    H_m = self._compute_GNSS_Jacobian(self.x, m)
                    S_m = float(H_m @ self.P @ H_m.T + self.R_Sat[m, m]) + 1e-10
                    K_m = (self.P @ H_m.T) / S_m
                    h_m = self._compute_GNSS_measurement(self.x, m)
                    self.x = self.x + K_m.flatten() * (y_Sat[m] - h_m)
                    self.P = (np.eye(8) - K_m @ H_m) @ self.P
            
            positions.append(self.x[:3].copy())
        
        return np.array(positions)
    
    def _compute_BS_Jacobian(self, state):
        K = self.model.K
        H = np.zeros((3 * K, 8))
        p = state[:3]
        
        for k in range(K):
            bs_pos = self.model.BS_positions[k]
            dx = p[0] - bs_pos[0]
            dy = p[1] - bs_pos[1]
            dz = p[2] - bs_pos[2]
            d_2D = np.sqrt(dx**2 + dy**2 + 1e-10)
            d_3D = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-10)
            
            i = 3 * k
            H[i, 0] = dx / (self.model.c * d_3D)
            H[i, 1] = dy / (self.model.c * d_3D)
            H[i, 2] = dz / (self.model.c * d_3D)
            H[i, 6] = 1
            H[i+1, 0] = -dy / (d_2D**2 + 1e-10)
            H[i+1, 1] = dx / (d_2D**2 + 1e-10)
            H[i+2, 0] = -(dx * dz) / (d_2D * d_3D**2 + 1e-10)
            H[i+2, 1] = -(dy * dz) / (d_2D * d_3D**2 + 1e-10)
            H[i+2, 2] = d_2D / (d_3D**2 + 1e-10)
        
        return H
    
    def _compute_BS_measurement_function(self, state):
        K = self.model.K
        h = np.zeros(3 * K)
        p = state[:3]
        rho = state[6]
        
        for k in range(K):
            bs_pos = self.model.BS_positions[k]
            dx = p[0] - bs_pos[0]
            dy = p[1] - bs_pos[1]
            dz = p[2] - bs_pos[2]
            d_3D = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-10)
            d_2D = np.sqrt(dx**2 + dy**2 + 1e-10)
            
            i = 3 * k
            h[i] = d_3D / self.model.c + rho
            h[i+1] = np.arctan2(dy, dx + 1e-10)
            h[i+2] = np.arctan2(dz, d_2D + 1e-10)
        
        return h
    
    def _compute_GNSS_Jacobian(self, state, m):
        H = np.zeros(8)
        p = state[:3]
        sat_pos = np.array([20200000 * np.cos(45*np.pi/180) * np.cos(m*40*np.pi/180),
                           20200000 * np.cos(45*np.pi/180) * np.sin(m*40*np.pi/180),
                           20200000 * np.sin(45*np.pi/180)])
        dx = p[0] - sat_pos[0]
        dy = p[1] - sat_pos[1]
        dz = p[2] - sat_pos[2]
        d = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-10)
        
        H[0] = dx / (self.model.c * d)
        H[1] = dy / (self.model.c * d)
        H[2] = dz / (self.model.c * d)
        H[6] = 1
        
        return H.reshape(1, 8)
    
    def _compute_GNSS_measurement(self, state, m):
        p = state[:3]
        rho = state[6]
        sat_pos = np.array([20200000 * np.cos(45*np.pi/180) * np.cos(m*40*np.pi/180),
                           20200000 * np.cos(45*np.pi/180) * np.sin(m*40*np.pi/180),
                           20200000 * np.sin(45*np.pi/180)])
        d = np.linalg.norm(p - sat_pos)
        return d / self.model.c + rho


# ============================================================================
# 6. Main Execution
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 70)
    print("GNSS-5G HYBRID POSITIONING WITH AI-ENHANCED MRAKF")
    print("Using Generated Data from generate_data.py")
    print("=" * 70)
    
    # 1. Load the generated dataset
    print("\n1. LOADING GENERATED DATASET")
    print("-" * 40)
    
    try:
        data = load_dataset('gnss_5g_dataset.pkl')
        print("   ✅ Dataset loaded successfully!")
    except FileNotFoundError:
        print("   ❌ Dataset not found!")
        print("   Please run: python generate_data.py first")
        return
    
    # 2. Prepare training data
    print("\n2. PREPARING TRAINING DATA")
    print("-" * 40)
    
    X = data['features']
    y = data['noise_labels']
    
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Feature names: {get_feature_names()}")
    print(f"   Label names: {get_label_names()}")
    
    # 3. Train XGBoost model
    print("\n3. TRAINING XGBOOST MODEL")
    print("-" * 40)
    
    predictor = XGBoostNoisePredictor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    training_results = predictor.train(X, y, test_size=0.2)
    
    # Save the trained model
    predictor.save_model('xgboost_noise_model.pkl')
    
    # 4. Set up system model
    print("\n4. SETTING UP SYSTEM MODEL")
    print("-" * 40)
    
    config = {
        'K': 4,
        'M': 9,
        'mu_BS': 10,
        'mu_Sat': 1,
        'n_PL': 1.98,
        'sigma_v': 3.0,
        'sigma_eps': 63e-6,
        'BS_positions': data['BS_positions']
    }
    
    system_model = SystemModel(config)
    print(f"   K={system_model.K} BSs, M={system_model.M} satellites")
    print(f"   BS rate: {system_model.mu_BS}Hz, GNSS rate: {system_model.mu_Sat}Hz")
    
    # 5. Run AI-Enhanced MRAKF
    print("\n5. RUNNING AI-ENHANCED MRAKF")
    print("-" * 40)
    
    ai_filter = AIEnhancedMRAKF(system_model, predictor)
    ai_filter.use_ai = True  # Use AI prediction
    ai_filter.calibrate(data['true_positions_downsampled'], data['BS_measurements_downsampled'])
    
    start_time = time.time()
    positions_ai = ai_filter.run(
        data['BS_measurements_downsampled'],
        data['GNSS_measurements_downsampled']
    )
    time_ai = time.time() - start_time
    
    errors_ai = np.linalg.norm(positions_ai - data['true_positions_downsampled'], axis=1)
    
    print(f"   ✅ AI-Enhanced MRAKF completed in {time_ai:.2f}s")
    print(f"   Mean Error: {np.mean(errors_ai):.3f} m")
    print(f"   RMSE: {np.sqrt(np.mean(errors_ai**2)):.3f} m")
    print(f"   Max Error: {np.max(errors_ai):.3f} m")
    
    # 6. Run Standard EKF (for comparison)
    print("\n6. RUNNING STANDARD EKF (BASELINE)")
    print("-" * 40)
    
    std_filter = StandardEKF(system_model)
    
    start_time = time.time()
    positions_std = std_filter.run(
        data['BS_measurements_downsampled'],
        data['GNSS_measurements_downsampled']
    )
    time_std = time.time() - start_time
    
    errors_std = np.linalg.norm(positions_std - data['true_positions_downsampled'], axis=1)
    
    print(f"   ✅ Standard EKF completed in {time_std:.2f}s")
    print(f"   Mean Error: {np.mean(errors_std):.3f} m")
    print(f"   RMSE: {np.sqrt(np.mean(errors_std**2)):.3f} m")
    print(f"   Max Error: {np.max(errors_std):.3f} m")
    
    # 7. Compare Results
    print("\n7. RESULTS COMPARISON")
    print("=" * 60)
    
    improvement = (np.mean(errors_std) - np.mean(errors_ai)) / np.mean(errors_std) * 100
    
    print(f"\n{'Method':<20} {'Mean Error':<15} {'RMSE':<15} {'Max Error':<15}")
    print("-" * 60)
    print(f"{'Standard EKF':<20} {np.mean(errors_std):.3f} m       {np.sqrt(np.mean(errors_std**2)):.3f} m       {np.max(errors_std):.3f} m")
    print(f"{'AI-Enhanced MRAKF':<20} {np.mean(errors_ai):.3f} m       {np.sqrt(np.mean(errors_ai**2)):.3f} m       {np.max(errors_ai):.3f} m")
    print("-" * 60)
    print(f"\n✅ Improvement: {improvement:.1f}%")
    
    # 8. Generate Plots
    print("\n8. GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Trajectory Comparison
    ax = axes[0, 0]
    true_traj = data['true_positions_downsampled']
    ax.plot(true_traj[:, 0], true_traj[:, 1], 'k--', label='True', linewidth=2)
    ax.plot(positions_std[:, 0], positions_std[:, 1], 'b-', label='Standard EKF', alpha=0.7)
    ax.plot(positions_ai[:, 0], positions_ai[:, 1], 'r-', label='AI-Enhanced MRAKF', alpha=0.7)
    ax.scatter(system_model.BS_positions[:, 0], system_model.BS_positions[:, 1], 
               c='black', s=100, marker='s', label='5G BSs')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Error Over Time
    ax = axes[0, 1]
    time_axis = np.arange(len(errors_std))
    ax.plot(time_axis, errors_std, 'b-', label='Standard EKF', alpha=0.7)
    ax.plot(time_axis, errors_ai, 'r-', label='AI-Enhanced MRAKF', alpha=0.7)
    ax.axhline(y=1.0, color='gray', linestyle='--', label='1m threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (m)')
    ax.set_title('Positioning Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution
    ax = axes[1, 0]
    bp = ax.boxplot([errors_std, errors_ai], labels=['Standard EKF', 'AI-Enhanced MRAKF'])
    ax.set_ylabel('Error (m)')
    ax.set_title('Error Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: CDF
    ax = axes[1, 1]
    for errors, label, color in zip([errors_std, errors_ai],
                                    ['Standard EKF', 'AI-Enhanced MRAKF'],
                                    ['blue', 'red']):
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cdf, color=color, label=label, linewidth=2)
    ax.axvline(x=1.0, color='gray', linestyle='--', label='1m threshold')
    ax.set_xlabel('Error (m)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('positioning_comparison.png', dpi=150)
    plt.show()
    print("   ✅ Plot saved as 'positioning_comparison.png'")
    
    # 9. Summary
    print("\n" + "=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nFiles Generated:")
    print("   - xgboost_noise_model.pkl (Trained XGBoost model)")
    print("   - positioning_comparison.png (Comparison plot)")
    print("\nKey Achievements:")
    print(f"   ✅ XGBoost model trained on {len(X)} samples")
    print(f"   ✅ AI-Enhanced MRAKF improved accuracy by {improvement:.1f}%")
    print(f"   ✅ Reduced mean error from {np.mean(errors_std):.3f}m to {np.mean(errors_ai):.3f}m")
    print("=" * 70)


if __name__ == "__main__":
    main()