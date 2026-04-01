import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
# ==========================================
# 0. LOAD RAW DATA
# ==========================================
# Assuming you downloaded the CSV from ChEMBL. 
# (Update the filename to match your exact downloaded file)
print("Loading raw ChEMBL data...")
raw_df = pd.read_csv('plasmodium_data3.csv', sep=';', low_memory=False)

# ==========================================
# 1. REMOVE MISSING VALUES
# ==========================================
print("Step 1: Filtering out missing chemical structures and test results...")

# We cannot train an AI if it doesn't know the molecular shape OR the test score
df_clean = raw_df.dropna(subset=['Smiles', 'pChEMBL Value']).copy()

# Force the pChEMBL column to be numeric (just in case pandas read it as text)
df_clean['pChEMBL Value'] = pd.to_numeric(df_clean['pChEMBL Value'], errors='coerce')

# Drop any rows that turned into NaN during the numeric conversion
df_clean = df_clean.dropna(subset=['pChEMBL Value'])

# ==========================================
# 2. RESOLVE DUPLICATES (The Professional Way)
# ==========================================
print("Step 2: Averaging scores for duplicate compounds...")

# Group the data by the Molecule ID and SMILES string.
# If a compound was tested 5 times by 5 different labs, this calculates its average potency.
df_reg = df_clean.groupby(['Molecule ChEMBL ID', 'Smiles'], as_index=False).agg({
    'pChEMBL Value': 'mean'
})

# ==========================================
# 3. FINAL CHECK
# ==========================================
print("\n--- Data Cleaning Complete ---")
print(f"Original raw rows: {raw_df.shape[0]}")
df_reg=df_reg.drop_duplicates(subset=['Molecule ChEMBL ID'])
print(f"Final clean, unique compounds ready for AI: {df_reg.shape[0]}")


# ==========================================
# 1. FEATURE EXTRACTION
# ==========================================
print("Step 1: Initializing modern fingerprint generator...")

# Initialize the modern Morgan Generator once (Radius=2, 2048 bits)
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_fingerprint(smiles):
    """Converts a SMILES string into a 2048-bit numpy array."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mfpgen.GetFingerprintAsNumPy(mol)
        return None
    except:
        return None

print("Step 2: Translating SMILES into fingerprints...")
df_reg['Fingerprint'] = df_reg['Smiles'].apply(get_fingerprint)

# Drop any complex molecules that RDKit couldn't process
df_reg = df_reg.dropna(subset=['Fingerprint'])

# ==========================================
# 2. DATA PREPARATION
# ==========================================
print("Step 3: Building the feature matrix (X) and target array (y)...")

X = np.stack(df_reg['Fingerprint'].values)
y = df_reg['pChEMBL Value'].values

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. MODEL TRAINING
# ==========================================
print("Step 4: Training the Random Forest Champion Model...")

# Initialize and train the baseline model that performed the best
champion_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
champion_model.fit(X_train, y_train)

# ==========================================
# 4. MODEL EVALUATION
# ==========================================
print("Step 5: Evaluating performance on unseen test data...")

y_pred = champion_model.predict(X_test)
final_r2 = r2_score(y_test, y_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- Final Model Metrics ---")
print(f"Testing R-squared: {final_r2:.3f}")
print(f"Testing RMSE:      {final_rmse:.3f} pChEMBL units")

# ==========================================
# 5. THE PREDICTION ENGINE
# ==========================================
def predict_new_drug(smiles, model):
    """Takes a raw SMILES string and returns the AI's predicted potency."""
    fp = get_fingerprint(smiles)
    if fp is None:
        return "Error: Invalid SMILES"
    
    # Reshape the 1D array into a 2D array (1 row, 2048 columns) for the model
    fp_reshaped = fp.reshape(1, -1)
    
    return model.predict(fp_reshaped)[0]

print("\nStep 6: Freezing and saving the AI model...")

# <-- This line creates the file you will drag and drop into GitHub!
joblib.dump(champion_model, 'malaria_rf_model.joblib', compress=9) 

print("Success! 'malaria_rf_model.joblib' is ready to be uploaded.")