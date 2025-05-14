#!/bin/bash
# Run the TabSyn evaluation script with Rice dataset

# Install required dependencies if needed
# Uncomment these lines to install dependencies
# pip install numpy pandas scikit-learn scipy icecream category_encoders

# Make the script executable
chmod +x run_tabsyn.py

# Run using the simplified wrapper
echo "Running TabSyn using the simplified wrapper..."
python run_tabsyn.py --dataset Rice --epochs 10 --seed 42 --verbose

# Run using the evaluation script
echo -e "\n\nRunning TabSyn using the evaluation script..."
python eval_tstr_discretized.py --models tabsyn --datasets Rice --n_rounds 1 --seed 42 --tabsyn_epochs 10 --output_prefix rice_tabsyn_eval --discretize

echo -e "\n\nAll done!"