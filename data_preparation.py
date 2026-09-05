import os
import pickle
import numpy as np

# Note: This script illustrates the preprocessing pipeline used to generate 
# Recurrence Plot (RP) images from single-lead ECG signals and save them as Pickle files.

def save_as_pickle(data, label, output_x_path, output_y_path):
    """
    Saves Recurrence Plot images and labels into Pickle format.
    
    Parameters:
    - data (numpy.ndarray): Array of Recurrence Plot images (Shape: [N, 299, 299, 3])
    - label (numpy.ndarray): Target labels (0: Normal, 1: Apnea)
    """
    print(f"Saving features to {output_x_path}...")
    with open(output_x_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Saving labels to {output_y_path}...")
    with open(output_y_path, 'wb') as g:
        pickle.dump(label, g, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Data successfully saved in Pickle format.")

if __name__ == '__main__':
    # Example usage:
    # Assuming `X_rp` contains Recurrence Plot matrix transformed from ECG segments
    # Shape of X_rp should be: (num_samples, 299, 299, 3)
    
    # x_Apnea_Data_inception.pickle & y_Apnea_Data_inception.pickle
    print("Preprocessing completed. Use generated pickle files in inception_apnea.py.")
