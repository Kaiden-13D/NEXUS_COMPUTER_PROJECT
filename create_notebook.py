import nbformat as nbf
import re

# List of files to be included in the notebook, in order
files_to_include = [
    "config.py",
    "data.py",
    "models.py",
    "client.py",
    "dcs.py",
    "clustering.py",
    "uav.py",
    "satellite.py",
    "metrics.py",
    "trainer.py"
]

# Create a new notebook
nb = nbf.v4.new_notebook()

# --- 1. Dependencies Cell ---
try:
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()

        
    install_code = "!pip install " + " ".join(f'"{r}"' for r in requirements)
    install_code = "!pip install datasets torch torchvision scikit-learn pandas tqdm"
    nb['cells'].append(nbf.v4.new_code_cell(install_code))
except FileNotFoundError:
    print("Could not find requirements.txt. Skipping dependency installation cell.")


# --- 2. Code Cells from Python Files ---
for file_path in files_to_include:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Remove if __name__ == "__main__": block by splitting
            content = content.split("if __name__ == '__main__':")[0]
            
            # Create the cell content with %%writefile
            cell_content = f"%%writefile {file_path}\n# --- Content from {file_path} ---\n\n{content.strip()}"
            
            nb['cells'].append(nbf.v4.new_code_cell(cell_content))
            
    except FileNotFoundError:
        print(f"Warning: Could not find file {file_path}. Skipping.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


# --- 3. Simulation Runner Cell ---
simulation_script_content = '''
from trainer import run_experiment
from config import CONFIG
run_experiment(CONFIG)
'''

runner_cell_content = f"""%%writefile run_simulation.py
{simulation_script_content}
"""
nb['cells'].append(nbf.v4.new_code_cell(runner_cell_content))


# --- 4. Write the Notebook to a File ---
output_notebook_path = "HPFL_Simulation_Generated.ipynb"
with open(output_notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully created notebook: {output_notebook_path}")