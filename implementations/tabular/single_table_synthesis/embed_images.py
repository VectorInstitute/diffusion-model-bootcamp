import base64
import os
import json
import re
from pathlib import Path

# Load the notebook
notebook_path = "tabsyn_implementation.ipynb"
figures_dir = "figures"

with open(notebook_path, "r") as f:
    notebook = json.load(f)

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Process each cell in the notebook
for cell in notebook["cells"]:
    if cell["cell_type"] == "markdown":
        for i, source in enumerate(cell["source"]):
            # Find image references
            img_matches = re.findall(r'<img src="figures/(.*?)"', source)
            
            # Replace image references with base64 embedded images
            for img_name in img_matches:
                img_path = os.path.join(figures_dir, img_name)
                if os.path.exists(img_path):
                    img_type = Path(img_path).suffix[1:]  # Get the extension without dot
                    base64_data = image_to_base64(img_path)
                    
                    # Replace the file reference with base64 data
                    img_tag = f'<img src="figures/{img_name}"'
                    base64_tag = f'<img src="data:image/{img_type};base64,{base64_data}"'
                    cell["source"][i] = cell["source"][i].replace(img_tag, base64_tag)

# Save the modified notebook
with open(notebook_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook updated with embedded images")
