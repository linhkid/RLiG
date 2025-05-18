#!/bin/bash
# Usage: ./run_all_datasets.sh [yes|no]
#
# Run evaluation on all datasets with or without discretization.
# Parameter:
#   yes - Run with discretization (default)
#   no  - Run without discretization
#
# Examples:
#   ./run_all_datasets.sh           # Run with discretization
#   ./run_all_datasets.sh yes       # Run with discretization (explicit)
#   ./run_all_datasets.sh no        # Run without discretization

# Create a timestamp for the main results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_DIR="results/$TIMESTAMP"
# Make sure the results directory exists
mkdir -p "results"
mkdir -p "$MAIN_DIR"

# Function to run evaluation and move results to dedicated folder
run_dataset() {
    local dataset_name="$1"
    local dataset_args="$2"
    local folder_name="$3"
    local use_discretize="$4"  # Add parameter for discretization option


    if [ -z "$folder_name" ]; then
        # Use dataset name as folder name if not specified
        folder_name="${dataset_name// /_}"
    fi
    
    # Determine discretization suffix for folder
    local disc_suffix=""
    local disc_option=""
    if [ "$use_discretize" = "no" ]; then
        disc_suffix="_nondiscrete"
        disc_option="--no-discretize"
        echo "Running WITHOUT discretization"
    else
        disc_suffix="_discrete"
        echo "Running WITH discretization (default)"
    fi
    
    # Always add nested cross-validation
    disc_option="$disc_option --nested-cv"
    echo "Using nested cross-validation"
    
    # Append discretization suffix to folder name
    folder_name="${folder_name}${disc_suffix}"
    
    echo "========================================================"
    echo "Running evaluation for dataset: $dataset_name"
    echo "Discretization: ${use_discretize:-yes}"
    echo "========================================================"
    
    # Create folder for results
    local results_dir="$MAIN_DIR/$folder_name"
    mkdir -p "$results_dir"
    
    # Run the evaluation
    if [ -z "$dataset_args" ]; then
        python eval_tstr_discretized.py --datasets "$dataset_name" $disc_option
    else
        python eval_tstr_discretized.py $dataset_args $disc_option
    fi
    
    # Move CSV results to the dataset folder
    echo "Moving results to $results_dir"
    
    # Move discretized or non-discretized results based on the flag
    if [ "$use_discretize" = "no" ]; then
        # For non-discretized results (they use raw_tstr prefix)
        mv raw_tstr_*.csv "$results_dir/" 2>/dev/null
        mv results/raw_tstr_*.csv "$results_dir/" 2>/dev/null
    else
        # For discretized results (they use disc_tstr prefix)
        mv disc_tstr_*.csv "$results_dir/" 2>/dev/null
        mv results/disc_tstr_*.csv "$results_dir/" 2>/dev/null
    fi
    
    # Delete the trainer_great folder if it exists
    if [ -d "trainer_great" ]; then
        echo "Removing trainer_great folder..."
        rm -rf "trainer_great"
    fi
    
    echo "Completed $dataset_name evaluation"
    echo ""
}

# Get discretization preference from command line argument (yes/no)
# Default to "yes" if not specified
DISCRETIZE=${1:-"no"}


# 1. Run Rice and Car datasets
run_dataset "Rice" "--datasets Rice" "rice" "$DISCRETIZE"

run_dataset "Car" "--datasets Car" "car" "$DISCRETIZE"

# 2. Run Adult dataset
run_dataset "Adult" "--datasets Adult" "adult" "$DISCRETIZE"

# 3. Run Chess dataset
run_dataset "Chess" "--datasets Chess" "chess" "$DISCRETIZE"

# 4. Run TicTacToe dataset
run_dataset "TicTacToe" "--datasets TicTacToe" "tictactoe" "$DISCRETIZE"

# 5. Run Letter Recognition dataset
run_dataset "letter_recog" "--datasets letter_recog" "letter_recognition" "$DISCRETIZE"

# 6. Run Magic dataset
run_dataset "Magic" "--datasets Magic" "magic" "$DISCRETIZE"

# 7. Run Nursery dataset
run_dataset "Nursery" "--datasets Nursery" "nursery" "$DISCRETIZE"

# 8. Run Room Occupancy dataset 
run_dataset "Room_Occupancy" "--datasets 'Room_Occupancy'" "room_occupancy" "$DISCRETIZE"

# 9. Run Maternal Health and Loan datasets
run_dataset "Maternal_Health" "--datasets 'Maternal_Health'" "maternal_health" "$DISCRETIZE"

run_dataset "Loan" "--datasets Loan" "loan" "$DISCRETIZE"

# 10. Run Credit dataset
run_dataset "Credit" "--datasets Credit" "credit" "$DISCRETIZE"

# 11. Run PokerHand dataset
run_dataset "PokerHand" "--datasets PokerHand" "pokerhand" "$DISCRETIZE"

# 12. Run Connect-4 dataset
run_dataset "Connect-4" "--datasets Connect-4" "connect4" "$DISCRETIZE"

# 13. Run NSL-KDD dataset
run_dataset "NSL-KDD" "--datasets NSL-KDD --local_datasets 'data/nsl-kdd/Full -d/KDDTrain20.arff'" "nsl_kdd" "$DISCRETIZE"

echo "All evaluations completed. Results are organized in: $MAIN_DIR (under the results/ directory)"

# Create a summary file with timestamps
echo "Dataset Evaluation Summary" > "$MAIN_DIR/summary.txt"
echo "Timestamp: $(date)" >> "$MAIN_DIR/summary.txt"
echo "=================================================" >> "$MAIN_DIR/summary.txt"
echo "Datasets processed:" >> "$MAIN_DIR/summary.txt"
ls -la "$MAIN_DIR" | grep -v "summary.txt" >> "$MAIN_DIR/summary.txt"
ls -la "$MAIN_DIR" | grep -v "summary.txt" >> "$MAIN_DIR/summary.txt"