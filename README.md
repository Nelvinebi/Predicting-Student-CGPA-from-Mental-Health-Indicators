Species Distribution Modeling with Environmental Variables (Synthetic Data)

This project demonstrates Species Distribution Modeling (SDM) using synthetic environmental data. It simulates environmental variables such as elevation, slope, temperature, precipitation, NDVI, distance to water, and human footprint, then generates species occurrence probabilities based on ecological response curves. Presence/absence data is sampled and can be used to train and evaluate machine learning models.

Features

Generates synthetic environmental variables on a grid (e.g., 20×20, 40×40).

Simulates species occurrence probabilities from ecological response curves.

Produces presence/absence labels based on habitat suitability.

Saves datasets in Excel (.xlsx) and CSV (.csv) formats.

Compatible with ML workflows (e.g., Logistic Regression, Random Forest).

Includes visualization of occurrence probability maps and presence points.

Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/Species-Distribution-Modeling-with-Environmental-Variables.git
cd Species-Distribution-Modeling-with-Environmental-Variables
pip install -r requirements.txt


Dependencies:

Python 3.8+

NumPy

Pandas

Scikit-learn

Matplotlib

OpenPyXL (for Excel export)

Usage

Run the script to generate synthetic SDM data:

python sdm_synthetic.py --rows 40 --cols 40 --seed 42 --out outputs


Arguments:

--rows: Grid rows (default: 40)

--cols: Grid columns (default: 40)

--seed: Random seed for reproducibility (default: 42)

--out: Output directory (default: outputs)

Example Output

outputs/sdm_synthetic_dataset.csv

outputs/sdm_synthetic_dataset.xlsx

outputs/map_true_prob.png (True probability map)

outputs/presence_overlay.png (Presence points overlay)

Dataset Columns

x, y: Normalized spatial coordinates

elevation, slope: Topographic variables

temperature, precipitation: Climatic variables

ndvi: Vegetation index (greenness)

dist_to_water: Distance to river feature

human_footprint: Anthropogenic pressure

occurrence_prob_true: True simulated suitability

presence: 1 = species present, 0 = absent

Applications

Teaching and demonstrating SDM workflows without real ecological data.

Benchmarking ML algorithms for ecological niche modeling.

Exploring ecological response functions and habitat suitability patterns.

Author

[Name: Agbozu Ebingiye Nelvin

 Github: https://github.com/Nelvinebi
 
 LinkedIn: *https://www.linkedin.com/in/agbozu-ebi/
]

License

This project is released under the MIT License. See LICENSE for details.
