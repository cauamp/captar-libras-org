import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Parse log data to extract epoch information
def parse_log_data(log_path):
    data = {}
    epoch = 0
    ctrl_wer = False
    
    with open(log_path, 'r') as file:
        for line in file:
            if 'Mean training loss:' in line:
                loss_value = float(line.split('Mean training loss:')[-1][:-2])
                data[epoch] = {'loss': loss_value}
            elif 'Ctrl WER:' in line:
                wer_value = float(line.split('Ctrl WER: ')[-1][:-2])
                if epoch not in data:
                    data[epoch] = {}
                data[epoch]['Ctrl WER'] = wer_value
                ctrl_wer = True
            elif 'Dev WER:' in line:
                wer_value = float(line.split('Dev WER:')[-1][:-2])
                if epoch not in data:
                    data[epoch] = {}
                data[epoch]['Dev WER'] = wer_value
                epoch += 1
    # Remove incomplete epochs
    data = {k: v for k, v in data.items() if 'loss' in v and 'Dev WER' in v}
    return data, ctrl_wer

# Save parsed data to CSV
def save_to_csv(data, csv_path, ctrl_wer=False):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    fields = ['epoch', 'loss', 'Dev WER']
    if ctrl_wer:
        fields.append('Ctrl WER')
        
    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch, values in data.items():
            if ctrl_wer:
                writer.writerow({'epoch': epoch, 'loss': values['loss'], 'Dev WER': values['Dev WER'], 'Ctrl WER': values['Ctrl WER']})
            else:
                writer.writerow({'epoch': epoch, 'loss': values['loss'], 'Dev WER': values['Dev WER']})

# Plot results
def plot_results(csv_path, log_dir, ctrl_wer=False):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 5))
    
    plt.suptitle(f'{train_dir}', fontsize=12, fontweight='bold')

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['loss'], marker='o', color='b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid()
    
    # Plot WER
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['Dev WER'], marker='o', color='r', label='Validation WER')
    if ctrl_wer:
        plt.plot(df['epoch'], df['Ctrl WER'], marker='o', color='g', label='Control WER')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('DEV WER')
    plt.title('Validation WER per Epoch ')
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(f'{log_dir}/LxW_sub.png')
    
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot Loss
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(df['epoch'], df['loss'], marker='o', color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid()
    plt.legend()

    # Create a second y-axis for WER
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('WER', color=color)
    ax2.plot(df['epoch'], df['Dev WER'], marker='o', color=color, label='Validation WER')
    if ctrl_wer:
        ax2.plot(df['epoch'], df['Ctrl WER'], marker='o', color='tab:green', label='Control WER')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid()

    plt.legend()
    plt.suptitle(f'{train_dir}', fontsize=12, fontweight='bold')
    plt.title('Training Loss and WER per Epoch')
    fig.tight_layout()
    plt.savefig(f'{log_dir}/LxW.png')

# Main function
if __name__ == '__main__':
    
    for train_dir in glob.glob('./*/'):
        log_txt_path = f'./{train_dir}/log.txt'
        
        if not os.path.exists(log_txt_path):
            continue
        
        log_dir = f'{train_dir}/log'
        log_csv_path = f'{log_dir}/log.csv'
        # Parse log data
        data, ctrl_wer = parse_log_data(log_txt_path)

        # Save data to CSV
        save_to_csv(data, log_csv_path, ctrl_wer)
        
        # Plot the results
        plot_results(log_csv_path, log_dir, ctrl_wer)
