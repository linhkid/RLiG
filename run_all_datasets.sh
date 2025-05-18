#!/bin/bash
# Usage: ./run_all_datasets.sh [yes|no]
#
# Run evaluation on all datasets with or without discretization.
# Parameter:
#   yes - Run with discretization
#   no  - Run without discretization (default behavior of the original script if no param passed)
#
# Examples:
#   ./run_all_datasets.sh           # Run without discretization (current script's default if $1 is empty)
#   ./run_all_datasets.sh yes       # Run with discretization
#   ./run_all_datasets.sh no        # Run without discretization (explicit)

# Create a timestamp for the main results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_DIR="results/$TIMESTAMP"
# Make sure the results directory exists
mkdir -p "results"
mkdir -p "$MAIN_DIR"

# Function to run evaluation and move results to dedicated folder
run_dataset() {
    local dataset_display_name="$1" # For echo and folder naming
    local python_script_args="$2"   # Arguments to pass to the python script
    local folder_name_prefix="$3"   # Prefix for the results folder for this dataset
    local use_discretize="$4"       # "yes" or "no" for discretization


    local folder_name_actual="${folder_name_prefix:-${dataset_display_name// /_}}" # Use prefix or generate from display_name

    local disc_suffix=""
    local disc_option=""
    if [ "$use_discretize" = "no" ]; then
        disc_suffix="_nondiscrete"
        disc_option="--no-discretize" # Python script should handle this flag
        # echo "Running WITHOUT discretization for $dataset_display_name" # Verbosity inside loop
    else
        disc_suffix="_discrete"
        # No specific option needed if --discretize is default true in python script,
        # or if the python script enables discretization by default when --no-discretize is absent.
        # echo "Running WITH discretization for $dataset_display_name"
    fi

    # Always add nested cross-validation as per original script's comments
    # The python script (eval_tstr_discretized.py) should have --nested-cv as an argument.
    #local common_options="$disc_option --nested-cv"

    folder_name_actual="${folder_name_actual}${disc_suffix}"

    echo "========================================================"
    echo "Running evaluation for dataset: $dataset_display_name"
    echo "Discretization: ${use_discretize:-yes}" # 'yes' if discretize is true, 'no' if false
    echo "Folder: $folder_name_actual"
    echo "========================================================"

    local results_dir="$MAIN_DIR/$folder_name_actual"
    mkdir -p "$results_dir"

    # Construct the command
    # The python script `eval_tstr_discretized.py` should be executable and in PATH, or provide full path.
    local full_command="python eval_tstr_discretized.py $python_script_args $common_options"

    echo "Executing: $full_command"
    eval $full_command # Use eval to correctly handle quotes/spaces in python_script_args

    echo "Moving results to $results_dir"

    # Determine results prefix based on discretization
    local results_prefix="disc_tstr" # Default prefix from python script if discretized
    if [ "$use_discretize" = "no" ]; then
        results_prefix="raw_tstr" # Prefix if not discretized
    fi

    # Move CSV results to the dataset folder
    # Using `find` for more robustly locating result files, as their exact output path from python script might vary slightly.
    # This assumes result files are created in the current directory or a predictable 'results' subdirectory by the python script.
    find . -maxdepth 1 -name "${results_prefix}_*.csv" -exec mv {} "$results_dir/" \; 2>/dev/null
    if [ -d "results" ]; then # Check if python script made its own "results" subfolder
      find results -maxdepth 1 -name "${results_prefix}_*.csv" -exec mv {} "$results_dir/" \; 2>/dev/null
    fi

    # Delete the trainer_great folder if it exists (from original script)
    if [ -d "trainer_great" ]; then
        echo "Removing trainer_great folder..."
        rm -rf "trainer_great"
    fi

    echo "Completed $dataset_display_name evaluation"
    echo ""
}

# Get discretization preference from command line argument
# Original script defaulted to "no" if $1 was empty.
DISCRETIZE_INPUT=${1:-"yes"}
# Normalize to "yes" or "no" for internal use
if [[ "$DISCRETIZE_INPUT" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    DISCRETIZE="yes"
else
    DISCRETIZE="no"
fi

echo "Overall Discretization setting for this batch run: $DISCRETIZE ('yes' for discrete, 'no' for non-discrete/raw)"
echo "Nested Cross-Validation will be enabled for all runs."
echo ""

# --- Modified UCI Dataset Calls ---
run_dataset "Adult" "--uci_ids 2 --datasets Adult" "adult" "$DISCRETIZE"
run_dataset "Car Evaluation" "--uci_ids 19 --datasets Car" "car" "$DISCRETIZE"
run_dataset "Chess (KRvKP)" "--uci_ids 22 --datasets Chess" "chess" "$DISCRETIZE"
run_dataset "Connect-4" "--uci_ids 26 --datasets Connect-4" "connect4" "$DISCRETIZE"
run_dataset "Credit Approval" "--uci_ids 27 --datasets Credit" "credit" "$DISCRETIZE"
run_dataset "Letter Recognition" "--uci_ids 59 --datasets letter_recog" "letter_recognition" "$DISCRETIZE"
run_dataset "MAGIC Gamma Telescope" "--uci_ids 159 --datasets Magic" "magic" "$DISCRETIZE"
run_dataset "Maternal Health Risk" "--uci_ids 863 --datasets Maternal_Health" "maternal_health" "$DISCRETIZE"
run_dataset "Nursery" "--uci_ids 76 --datasets Nursery" "nursery" "$DISCRETIZE"
run_dataset "Room Occupancy" "--uci_ids 864 --datasets Room_Occupancy" "room_occupancy" "$DISCRETIZE"
run_dataset "Poker Hand" "--uci_ids 158 --datasets PokerHand" "pokerhand" "$DISCRETIZE"
run_dataset "Rice (Cammeo and Osmancik)" "--uci_ids 545 --datasets Rice" "rice" "$DISCRETIZE"
run_dataset "Tic-Tac-Toe Endgame" "--uci_ids 101 --datasets TicTacToe" "tictactoe" "$DISCRETIZE"

## --- Datasets from original script using the previous argument style ---
## (Assuming eval_tstr_discretized.py can handle these names, possibly as predefined local datasets or special UCI cases)
#run_dataset "Rice" "--datasets Rice" "rice" "$DISCRETIZE"
#run_dataset "TicTacToe" "--datasets TicTacToe" "tictactoe" "$DISCRETIZE"


# --- Local Datasets (Keeping existing command structure from original script) ---
# Ensure your python script (eval_tstr_discretized.py) correctly uses the --datasets name
# in conjunction with the --local_datasets path.
run_dataset "Loan (Local)" "--datasets Loan --local_datasets 'data/loan_approval_dataset.csv'" "loan" "$DISCRETIZE"
run_dataset "NSL-KDD (Local)" "--datasets NSL-KDD --local_datasets 'data/nsl-kdd/Full Data/KDDTrain+.arff'" "nsl_kdd" "$DISCRETIZE"
# If you intended to use KDDTrain20.arff as in the original script comment:
# run_dataset "NSL-KDD (Local)" "--datasets NSL-KDD --local_datasets 'data/nsl-kdd/Full Data/KDDTrain20.arff'" "nsl_kdd" "$DISCRETIZE"


echo "All specified dataset evaluations completed."
echo "Results are organized in subfolders within: $MAIN_DIR"

# Create a summary file with timestamps
echo "Dataset Evaluation Summary" > "$MAIN_DIR/summary.txt"
echo "Run Timestamp: $TIMESTAMP" >> "$MAIN_DIR/summary.txt"
echo "Discretization Mode: $DISCRETIZE" >> "$MAIN_DIR/summary.txt"
echo "=================================================" >> "$MAIN_DIR/summary.txt"
echo "Datasets processed (see subfolders):" >> "$MAIN_DIR/summary.txt"
# List subfolders in MAIN_DIR, excluding the summary file itself.
ls -1 "$MAIN_DIR" | grep -v "summary.txt" >> "$MAIN_DIR/summary.txt"

echo "Summary file created at $MAIN_DIR/summary.txt"