import subprocess
import sys

sub_folders = ["docvqa" ,"textcap" ,"textvqa",  "dude",  "sroie", "infographicsvqa" , "flickr30k", "gqa" ,"openimages" ,"vsr", "cub"]

if len(sys.argv) != 2:
    print("Usage: python script.py <system_input>")
    sys.exit(1)

system_input = sys.argv[1]

accuracy_table = {}
line_counts = {}

# Loop through sub-folders and run the command for each one
for sub_folder in sub_folders:
    command = f"python playground/data/eval/REC/calculation.py --results_dir ./results/viscot/detection/{sub_folder}/{system_input}.jsonl"
    print(command)

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        output_lines = result.stdout.split('\n')

        # Count the number of lines in the command's output
        line_count = len(output_lines) - 1
        line_counts[sub_folder] = line_count
        if len(output_lines) >= 2:
            accuracy_line = output_lines[-2]  # Get the last but one line
            accuracy_parts = accuracy_line.split()
            if len(accuracy_parts) > 0:
                accuracy = float(accuracy_parts[0]) * 100
                accuracy_table[sub_folder] = accuracy
                print(f"Accuracy for {sub_folder}: {accuracy:.2f}, Error format: {line_count}")
            else:
                print(f"Accuracy for {sub_folder} not found in output.")
        else:
            print(f"No accuracy information found in output for {sub_folder}. Error format: {line_count}")
    except subprocess.CalledProcessError as e:
        print(f"Error running script for {sub_folder}: {e}")

# Calculate the average accuracy
all_accuracies = [float(accuracy) for accuracy in accuracy_table.values() if accuracy]
if all_accuracies:
    average_accuracy = sum(all_accuracies) / len(all_accuracies)
    print(f"\nAverage Accuracy: {average_accuracy:.3f}")

# Print the accuracy table
print("\nAccuracy Table:")
for sub_folder, accuracy in accuracy_table.items():
    print(f"{sub_folder}: {accuracy}")

# Print the line counts
print("\nError format:")
for sub_folder, count in line_counts.items():
    print(f"{sub_folder}: {count}")

