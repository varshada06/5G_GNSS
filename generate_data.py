# generate_data.py
# Data Generation for GNSS-5G Hybrid Positioning Project

import numpy as np
import pandas as pd
import pickle
import os
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. System Model Classes
# ============================================================================

class GNSS5GSystemModel:
    """Implements the system model from Sections III-A to III-D"""
    
    def __init__(self, config):
        self.config = config
        self.c = 299792458.0  # Speed of light
        
        # BS configuration
        self.K = config.get('K', 4)  # Number of BSs
        self.N_T = config.get('N_T', 36)  # Transmit antennas
        self.N_R = config.get('N_R', 16)  # Receive antennas
        
        # GNSS configuration
        self.M = config.get('M', 9)  # Number of visible satellites
        
        # Data rates
        self.mu_BS = config.get('mu_BS', 10)  # BS measurement rate (samples/s)
        self.mu_Sat = config.get('mu_Sat', 1)  # GNSS measurement rate (samples/s)
        self.R_a = self.mu_BS // self.mu_Sat  # Rate ratio
        
        # Path loss parameters (UMi-street canyon)
        self.n_PL = config.get('n_PL', 1.98)  # Path loss exponent
        self.sigma_SF = config.get('sigma_SF', 3.1)  # Shadow fading (dB)
        self.f_c = config.get('f_c', 28e9)  # Carrier frequency (28 GHz)
        
        # UE motion parameters
        self.sigma_v = config.get('sigma_v', 3.0)  # STD of velocity
        self.sigma_eps = config.get('sigma_eps', 63e-6)  # STD of clock skew noise
        
        # Clock model parameters
        self.omega = config.get('omega', 1.0)  # AR coefficient
        
        # Initialize BS positions
        self.BS_positions = self._generate_BS_positions()
        self.satellite_positions = self._generate_satellite_positions()
        
    def _generate_BS_positions(self):
        """Generate non-collinear BS positions"""
        d_BS_BS = self.config.get('d_BS_BS', 50)
        d_BS_R = self.config.get('d_BS_R', 20)
        
        positions = []
        for i in range(self.K):
            x = i * d_BS_BS
            y = d_BS_R * (1 if i % 2 == 0 else -1)
            z = 10  # BS height
            positions.append([x, y, z])
        return np.array(positions)
    
    def _generate_satellite_positions(self):
        """Generate satellite positions"""
        np.random.seed(42)
        sat_positions = []
        for m in range(self.M):
            theta = np.random.uniform(10, 80) * np.pi / 180
            phi = np.random.uniform(0, 2 * np.pi)
            r = 20200000  # GPS orbit radius
            
            x = r * np.cos(theta) * np.cos(phi)
            y = r * np.cos(theta) * np.sin(phi)
            z = r * np.sin(theta)
            sat_positions.append([x, y, z])
        return np.array(sat_positions)
    
    def generate_GNSS_measurements(self, state, epoch):
        """Generate GNSS pseudorange measurements"""
        p = state[:3]
        rho = state[6]
        
        measurements = []
        for m in range(self.M):
            sat_pos = self.satellite_positions[m]
            true_range = np.linalg.norm(p - sat_pos)
            sigma_Sat = self.config.get('sigma_Sat', 0.5)
            noise = np.random.normal(0, sigma_Sat)
            measurement = true_range / self.c + rho + noise
            measurements.append(measurement)
        
        return np.array(measurements)
    
    def generate_BS_measurements(self, state, epoch):
        """Generate 5G BS measurements"""
        p = state[:3]
        rho = state[6]
        
        measurements = []
        for k in range(self.K):
            bs_pos = self.BS_positions[k]
            dx = p[0] - bs_pos[0]
            dy = p[1] - bs_pos[1]
            dz = p[2] - bs_pos[2]
            
            d_3D = np.sqrt(dx**2 + dy**2 + dz**2)
            d_2D = np.sqrt(dx**2 + dy**2)
            
            true_TOA = d_3D / self.c + rho
            true_azimuth = np.arctan2(dy, dx)
            true_elevation = np.arctan2(dz, d_2D + 1e-10)
            
            # Calculate noise STDs using CRLB
            sigma_TOA, sigma_az, sigma_el = self._calculate_measurement_noise_STD(d_3D)
            
            noise_TOA = np.random.normal(0, sigma_TOA)
            noise_az = np.random.normal(0, sigma_az)
            noise_el = np.random.normal(0, sigma_el)
            
            measurement = np.array([true_TOA + noise_TOA, 
                                   true_azimuth + noise_az, 
                                   true_elevation + noise_el])
            measurements.append(measurement)
        
        return np.array(measurements)
    
    def _calculate_measurement_noise_STD(self, d):
        """Calculate noise STDs based on distance"""
        CRLB0_TOA = 1e-16
        CRLB0_az = 1e-4
        CRLB0_el = 1e-4
        
        sigma_TOA = np.sqrt(CRLB0_TOA * (d ** self.n_PL))
        sigma_az = np.sqrt(CRLB0_az * (d ** self.n_PL))
        sigma_el = np.sqrt(CRLB0_el * (d ** self.n_PL))
        
        return sigma_TOA, sigma_az, sigma_el
    
    def get_state_transition_matrix(self, dt):
        """Get state transition matrix F"""
        F = np.eye(8)
        F[0:3, 3:6] = dt * np.eye(3)
        F[6, 7] = dt
        F[7, 7] = self.omega
        return F
    
    def get_process_noise_covariance(self, dt):
        """Get process noise covariance matrix Q"""
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
# 2. Data Generator
# ============================================================================

class DataGenerator:
    """Generate simulation data with true noise labels"""
    
    def __init__(self, system_model):
        self.model = system_model
        
    def generate_trajectory(self, duration=60, dt=0.1):
        """Generate UE trajectory"""
        N_steps = int(duration / dt)
        trajectory = np.zeros((N_steps, 3))
        
        for i in range(N_steps):
            t = i * dt
            trajectory[i, 0] = 5 * t  # x position
            trajectory[i, 1] = 0      # y position
            trajectory[i, 2] = 1.5    # z position
        
        return trajectory
    
    def generate_complete_dataset(self, duration=60, dt=0.1):
        """Generate complete dataset with all measurements and labels"""
        trajectory = self.generate_trajectory(duration, dt)
        N_steps = len(trajectory)
        
        features = []
        noise_labels = []
        BS_measurements = []
        GNSS_measurements = []
        true_positions = []
        
        clock_offset = 0
        clock_skew = 0
        
        for i in range(N_steps):
            p = trajectory[i]
            
            clock_offset += clock_skew * dt
            clock_skew += np.random.normal(0, self.model.sigma_eps * np.sqrt(dt))
            
            state = np.zeros(8)
            state[:3] = p
            state[3:6] = np.array([5, 0, 0])
            state[6] = clock_offset
            state[7] = clock_skew
            
            BS_meas = self.model.generate_BS_measurements(state, i)
            GNSS_meas = self.model.generate_GNSS_measurements(state, i)
            
            BS_measurements.append(BS_meas)
            GNSS_measurements.append(GNSS_meas)
            true_positions.append(p)
            
            # Extract features for training
            for k in range(self.model.K):
                bs_pos = self.model.BS_positions[k]
                d = np.linalg.norm(p - bs_pos)
                
                features.append([
                    d,
                    k,
                    p[0], p[1], p[2],
                    state[3], state[4], state[5],
                    i / N_steps,
                    np.sin(2 * np.pi * i / N_steps),
                    np.cos(2 * np.pi * i / N_steps),
                ])
                
                sigma_TOA, sigma_az, sigma_el = self.model._calculate_measurement_noise_STD(d)
                noise_labels.append([sigma_TOA, sigma_az, sigma_el])
        
        # Downsample for GNSS rate
        R_a = self.model.R_a
        BS_measurements_downsampled = BS_measurements[::R_a]
        GNSS_measurements_downsampled = GNSS_measurements[::R_a]
        true_positions_downsampled = true_positions[::R_a]
        
        return {
            'features': np.array(features),
            'noise_labels': np.array(noise_labels),
            'BS_measurements': np.array(BS_measurements),
            'GNSS_measurements': np.array(GNSS_measurements),
            'BS_measurements_downsampled': np.array(BS_measurements_downsampled),
            'GNSS_measurements_downsampled': np.array(GNSS_measurements_downsampled),
            'true_positions': np.array(true_positions),
            'true_positions_downsampled': np.array(true_positions_downsampled),
            'trajectory': trajectory,
            'BS_positions': self.model.BS_positions,
            'config': self.model.config
        }


# ============================================================================
# 3. Data Loading and Saving
# ============================================================================

def save_dataset(data, filename='gnss_5g_dataset.pkl'):
    """Save dataset to file"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Dataset saved to {filename}")
    print(f"  - Features shape: {data['features'].shape}")
    print(f"  - Noise labels shape: {data['noise_labels'].shape}")
    print(f"  - BS measurements: {data['BS_measurements'].shape}")
    print(f"  - GNSS measurements: {data['GNSS_measurements'].shape}")

def load_dataset(filename='gnss_5g_dataset.pkl'):
    """Load dataset from file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Dataset loaded from {filename}")
    return data

def save_csv_dataset(data, prefix='gnss_5g'):
    """Save dataset to CSV files"""
    # Features and labels
    feature_names = ['distance', 'bs_index', 'pos_x', 'pos_y', 'pos_z', 
                     'vel_x', 'vel_y', 'vel_z', 'time_norm', 'sin_time', 'cos_time']
    label_names = ['sigma_toa', 'sigma_azimuth', 'sigma_elevation']
    
    df_features = pd.DataFrame(data['features'], columns=feature_names)
    df_labels = pd.DataFrame(data['noise_labels'], columns=label_names)
    
    df_features.to_csv(f'{prefix}_features.csv', index=False)
    df_labels.to_csv(f'{prefix}_labels.csv', index=False)
    
    # Measurements (saved as separate files due to size)
    np.savetxt(f'{prefix}_bs_measurements.csv', 
               data['BS_measurements_downsampled'].reshape(-1, data['BS_measurements_downsampled'].shape[1]*data['BS_measurements_downsampled'].shape[2]),
               delimiter=',')
    np.savetxt(f'{prefix}_gnss_measurements.csv', 
               data['GNSS_measurements_downsampled'], 
               delimiter=',')
    np.savetxt(f'{prefix}_true_positions.csv', 
               data['true_positions_downsampled'], 
               delimiter=',')
    
    print(f"CSV files saved with prefix '{prefix}'")


# ============================================================================
# 4. Data Visualization
# ============================================================================

def visualize_dataset(data, save_plots=True):
    """Visualize the generated dataset"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Trajectory
    ax = axes[0, 0]
    traj = data['trajectory']
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='UE Trajectory')
    bs_pos = data['BS_positions']
    ax.scatter(bs_pos[:, 0], bs_pos[:, 1], c='red', s=100, marker='s', label='5G BSs')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('UE Trajectory and BS Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Distance to BS over time
    ax = axes[0, 1]
    features = data['features']
    for k in range(4):
        mask = features[:, 1] == k
        if np.any(mask):
            distances = features[mask, 0]
            ax.plot(distances, label=f'BS{k+1}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance (m)')
    ax.set_title('UE-BS Distance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Noise STDs vs Distance
    ax = axes[0, 2]
    distances = features[:, 0]
    sigma_toa = data['noise_labels'][:, 0]
    ax.scatter(distances, sigma_toa, s=1, alpha=0.5, label='TOA')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Noise STD')
    ax.set_title('Noise STD vs Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Noise distribution
    ax = axes[1, 0]
    ax.hist(data['noise_labels'][:, 0], bins=50, alpha=0.5, label='TOA')
    ax.hist(data['noise_labels'][:, 1], bins=50, alpha=0.5, label='Azimuth')
    ax.hist(data['noise_labels'][:, 2], bins=50, alpha=0.5, label='Elevation')
    ax.set_xlabel('Noise STD')
    ax.set_ylabel('Frequency')
    ax.set_title('Noise Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Correlation matrix
    ax = axes[1, 1]
    feature_names = ['Dist', 'BS', 'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']
    corr_data = data['features'][:1000, :8]  # Use subset
    corr_matrix = np.corrcoef(corr_data.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_yticklabels(feature_names)
    ax.set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=ax)
    
    # 6. Measurement counts
    ax = axes[1, 2]
    ax.bar(['BS\nMeasurements', 'GNSS\nMeasurements', 'Features', 'Labels'],
           [len(data['BS_measurements_downsampled']), 
            len(data['GNSS_measurements_downsampled']),
            len(data['features']),
            len(data['noise_labels'])])
    ax.set_ylabel('Count')
    ax.set_title('Dataset Statistics')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('dataset_visualization.png', dpi=150)
    plt.show()
    print("Visualization complete")


# ============================================================================
# 5. Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GNSS-5G Dataset Generation")
    print("=" * 70)
    
    # Configuration
    config = {
        'K': 4,
        'M': 9,
        'N_R': 16,
        'N_T': 36,
        'mu_BS': 10,
        'mu_Sat': 1,
        'n_PL': 1.98,
        'sigma_v': 3.0,
        'sigma_eps': 63e-6,
        'sigma_Sat': 0.5,
        'd_BS_BS': 50,
        'd_BS_R': 20,
        'f_c': 28e9,
        'duration': 60,
        'dt': 0.1
    }
    
    print("\n1. Initializing system model...")
    system_model = GNSS5GSystemModel(config)
    print(f"   - {system_model.K} BSs, {system_model.M} satellites")
    print(f"   - BS rate: {system_model.mu_BS} Hz, GNSS rate: {system_model.mu_Sat} Hz")
    
    print("\n2. Generating dataset...")
    data_gen = DataGenerator(system_model)
    data = data_gen.generate_complete_dataset(
        duration=config['duration'],
        dt=config['dt']
    )
    
    print(f"\n3. Dataset summary:")
    print(f"   - Features: {data['features'].shape[0]} samples, {data['features'].shape[1]} features")
    print(f"   - Noise labels: {data['noise_labels'].shape[0]} samples, {data['noise_labels'].shape[1]} outputs")
    print(f"   - BS measurements: {data['BS_measurements'].shape[0]} epochs")
    print(f"   - GNSS measurements: {data['GNSS_measurements'].shape[0]} epochs")
    print(f"   - True positions: {data['true_positions'].shape[0]} epochs")
    
    print("\n4. Saving dataset...")
    save_dataset(data, 'gnss_5g_dataset.pkl')
    save_csv_dataset(data)
    
    print("\n5. Visualizing dataset...")
    visualize_dataset(data, save_plots=True)
    
    print("\n" + "=" * 70)
    print("Data Generation Complete!")
    print("Files created:")
    print("  - gnss_5g_dataset.pkl (Pickle format)")
    print("  - gnss_5g_features.csv")
    print("  - gnss_5g_labels.csv")
    print("  - gnss_5g_bs_measurements.csv")
    print("  - gnss_5g_gnss_measurements.csv")
    print("  - gnss_5g_true_positions.csv")
    print("  - dataset_visualization.png")
    print("=" * 70)