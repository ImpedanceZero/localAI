import glob
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Step 1: Parse the log files
def parse_anything_llm_logs(log_files):
    client_requests = []  # List of (timestamp, user_id, prompt, workspace) for client requests
    errors = []          # List of (timestamp, error_type) for errors
    prompt_eval_tps = [] # List of (timestamp, model, tps) for prompt eval TPS
    generation_tps = []  # List of (timestamp, model, tps) for generation TPS
    ttft = []            # List of (timestamp, model, ttft_ms) for TTFT
    total_latency = []   # List of (timestamp, model, latency_ms) for total response latency

    # Regex patterns for parsing log lines
    # Pattern for CLIENT_REQUEST in anything_llm_test.log
    client_request_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*CLIENT_REQUEST - RequestID: (\d+), UserID: (\d+), Prompt: \'(.*)\', Workspace: \'(.*)\''
    )
    # Pattern for timeout errors in anything_llm_test.log
    timeout_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - AnythingLLMTest - ERROR - User \d+ \(Iteration \d+\): Request failed: HTTPConnectionPool'
    )
    # Patterns for container logs (now with timestamps)
    post_pattern = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\] srv  log_server_r: request: POST /v1/chat/completions \S+ (\d{3})'
    )
    crash_pattern = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\] terminate called without an active exception'
    )
    http_503_pattern = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\] srv  log_server_r: request: GET /v1/models \S+ 503'
    )
    prompt_eval_pattern = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\] prompt eval time =\s*([\d.]+) ms(?:.*?\(.*?,\s*([\d.]+) tokens per second\))?'
    )
    eval_time_pattern = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]\s*eval time\s*=\s*([\d.]+)\s*ms\s*/\s*\d+\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)'
    )
    total_time_pattern = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]\s*total time\s*=\s*([\d.]+)\s*ms\s*/\s*\d+\s*tokens'
    )
    # Debug patterns to detect any prompt eval time, eval time, or total time lines
    prompt_eval_any_pattern = re.compile(
        r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] prompt eval time ='
    )
    eval_time_any_pattern = re.compile(
        r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]\s*eval time\s*='
    )
    total_time_any_pattern = re.compile(
        r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]\s*total time\s*='
    )

    # Debug: Print the list of log files being processed
    main_log_files = [f for f in log_files if 'anything_llm_test.log' in f]
    container_log_files = [f for f in log_files if 'llama_cpp_metrics_' in f]
    print(f"Processing main log files: {main_log_files}")
    print(f"Processing container log files: {container_log_files}")

    # Counters and lists for debug logging
    prompt_eval_lines_found = 0
    eval_time_lines_found = 0
    total_time_lines_found = 0
    unmatched_prompt_eval_lines = []
    unmatched_eval_time_lines = []
    unmatched_total_time_lines = []

    # Process each log file
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                # Parse CLIENT_REQUEST messages from anything_llm_test.log
                if 'anything_llm_test.log' in log_file:
                    client_request_match = client_request_pattern.match(line)
                    if client_request_match:
                        timestamp_str, request_id, user_id, prompt, workspace = client_request_match.groups()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        # Determine the model based on the workspace
                        model = 'Codestral' if 'codesage-code-assistant' in workspace else 'Mistral'
                        client_requests.append((timestamp, user_id, prompt, workspace, model))
                        continue

                    # Parse timeout errors from anything_llm_test.log
                    timeout_match = timeout_pattern.match(line)
                    if timeout_match:
                        timestamp_str, = timeout_match.groups()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        errors.append((timestamp, 'timeout'))
                        continue

                # Parse container logs from llama_cpp_metrics_*.log
                if 'llama_cpp_metrics_' in log_file:
                    # Determine the model based on the log file name
                    model = 'Codestral' if 'codestral' in log_file else 'Mistral'

                    # Debug: Check for any prompt eval time, eval time, or total time lines
                    prompt_eval_any_match = prompt_eval_any_pattern.match(line)
                    if prompt_eval_any_match:
                        prompt_eval_lines_found += 1
                        if not prompt_eval_pattern.match(line):
                            if len(unmatched_prompt_eval_lines) < 5:  # Limit to 5 examples
                                unmatched_prompt_eval_lines.append(line.strip())

                    eval_time_any_match = eval_time_any_pattern.match(line)
                    if eval_time_any_match:
                        eval_time_lines_found += 1
                        if not eval_time_pattern.match(line):
                            if len(unmatched_eval_time_lines) < 5:  # Limit to 5 examples
                                unmatched_eval_time_lines.append(line.strip())

                    total_time_any_match = total_time_any_pattern.match(line)
                    if total_time_any_match:
                        total_time_lines_found += 1
                        if not total_time_pattern.match(line):
                            if len(unmatched_total_time_lines) < 5:  # Limit to 5 examples
                                unmatched_total_time_lines.append(line.strip())

                    # Parse POST requests (not used for plotting, but kept for metrics)
                    post_match = post_pattern.match(line)
                    if post_match:
                        timestamp_str, status = post_match.groups()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        # Not appending to requests list since we're plotting client requests
                        continue

                    # Parse crash errors
                    crash_match = crash_pattern.match(line)
                    if crash_match:
                        timestamp_str, = crash_match.groups()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        errors.append((timestamp, 'crash'))
                        continue

                    # Parse HTTP 503 errors
                    http_503_match = http_503_pattern.match(line)
                    if http_503_match:
                        timestamp_str, = http_503_match.groups()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        errors.append((timestamp, 'http_503'))
                        continue

                    # Parse prompt eval time
                    prompt_eval_match = prompt_eval_pattern.match(line)
                    if prompt_eval_match:
                        timestamp_str, prompt_eval_time, prompt_tps = prompt_eval_match.groups()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        if prompt_tps:  # Only append if TPS is present
                            prompt_eval_tps.append((timestamp, model, float(prompt_tps)))
                        ttft.append((timestamp, model, float(prompt_eval_time)))
                        continue

                    # Parse eval time
                    eval_time_match = eval_time_pattern.match(line)
                    if eval_time_match:
                        timestamp_str, eval_time, gen_tps = eval_time_match.groups()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        if gen_tps:  # Only append if TPS is present
                            generation_tps.append((timestamp, model, float(gen_tps)))
                        continue

                    # Parse total time
                    total_time_match = total_time_pattern.match(line)
                    if total_time_match:
                        timestamp_str, total_time = total_time_match.groups()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        total_latency.append((timestamp, model, float(total_time)))
                        continue

    # Debug logging to check the number of parsed data points
    print(f"Parsed {len(client_requests)} client requests")
    if len(client_requests) < 740:  # Expected number based on series3_input.csv
        print(f"Warning: Expected 740 client requests, but only {len(client_requests)} were parsed. The test run may be incomplete.")
    print(f"Parsed {len(errors)} errors (crash: {sum(1 for _, t in errors if t == 'crash')}, HTTP 503: {sum(1 for _, t in errors if t == 'http_503')}, timeout: {sum(1 for _, t in errors if t == 'timeout')})")
    print(f"Found {prompt_eval_lines_found} prompt eval time lines in total")
    if prompt_eval_lines_found > 0 and len(prompt_eval_tps) == 0:
        print("Some prompt eval time lines were not parsed. Here are up to 5 examples of unmatched lines:")
        for i, line in enumerate(unmatched_prompt_eval_lines, 1):
            print(f"Unmatched prompt eval line {i}: {line}")
    print(f"Found {eval_time_lines_found} eval time lines in total")
    if eval_time_lines_found > 0 and len(generation_tps) == 0:
        print("Some eval time lines were not parsed. Here are up to 5 examples of unmatched lines:")
        for i, line in enumerate(unmatched_eval_time_lines, 1):
            print(f"Unmatched eval time line {i}: {line}")
    print(f"Found {total_time_lines_found} total time lines in total")
    if total_time_lines_found > 0 and len(total_latency) == 0:
        print("Some total time lines were not parsed. Here are up to 5 examples of unmatched lines:")
        for i, line in enumerate(unmatched_total_time_lines, 1):
            print(f"Unmatched total time line {i}: {line}")
    print(f"Parsed {len(prompt_eval_tps)} Prompt Eval TPS entries")
    print(f"Parsed {len(generation_tps)} Generation TPS entries")
    print(f"Parsed {len(ttft)} TTFT entries")
    print(f"Parsed {len(total_latency)} Total Response Latency entries")

    return client_requests, errors, prompt_eval_tps, generation_tps, ttft, total_latency

# Step 2: Parse the nvidia_smi_metrics.csv file
def parse_nvidia_smi_metrics(csv_file):
    df = pd.read_csv(csv_file)
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Step 3: Generate the scatter plots
def generate_scatter_plots(client_requests, errors, prompt_eval_tps, generation_tps, ttft, total_latency, nvidia_df):
    # Create a figure with five subplots, sharing the x-axis
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 20), sharex=True, gridspec_kw={'height_ratios': [2, 2, 2, 2, 2]})

    # --- First Subplot: Requests, Errors, GPU Stats ---
    # Plot client requests (on ax1), differentiated by model
    codestral_requests = [(ts, -5) for ts, _, _, _, model in client_requests if model == 'Codestral']
    mistral_requests = [(ts, -10) for ts, _, _, _, model in client_requests if model == 'Mistral']
    
    if codestral_requests:
        codestral_times, codestral_y = zip(*codestral_requests)
        ax1.scatter(codestral_times, codestral_y, color='blue', marker='o', label='Codestral Requests', s=50)
    
    if mistral_requests:
        mistral_times, mistral_y = zip(*mistral_requests)
        ax1.scatter(mistral_times, mistral_y, color='green', marker='o', label='Mistral Requests', s=50)

    # Plot errors (on ax1) only if data is present
    crash_errors = [(ts, -15) for ts, error_type in errors if error_type == 'crash']
    http_503_errors = [(ts, -20) for ts, error_type in errors if error_type == 'http_503']
    timeout_errors = [(ts, -25) for ts, error_type in errors if error_type == 'timeout']
    
    # Dynamically set y-axis ticks and labels based on whether errors are present
    y_ticks = [-10, -5, 0, 20, 40, 60, 80, 100]
    y_labels = ['Mistral Requests', 'Codestral Requests', '0', '20', '40', '60', '80', '100']
    error_y_offset = 0

    if crash_errors:
        crash_times, crash_y = zip(*crash_errors)
        ax1.scatter(crash_times, crash_y, color='red', marker='x', label='Crash Errors', s=100)
        y_ticks.insert(0, -15)
        y_labels.insert(0, 'Crash Errors')
        error_y_offset += 5

    if http_503_errors:
        http_503_times, http_503_y = zip(*http_503_errors)
        ax1.scatter(http_503_times, http_503_y, color='red', marker='^', label='HTTP 503 Errors', s=100)
        y_ticks.insert(0, -20)
        y_labels.insert(0, 'HTTP 503 Errors')
        error_y_offset += 5

    if timeout_errors:
        timeout_times, timeout_y = zip(*timeout_errors)
        ax1.scatter(timeout_times, timeout_y, color='red', marker='*', label='Timeout Errors', s=100)
        y_ticks.insert(0, -25)
        y_labels.insert(0, 'Timeout Errors')

    # Plot GPU stats (on ax1)
    ax1.scatter(nvidia_df['timestamp'], nvidia_df['gpu_utilization'], color='orange', marker='.', label='GPU Utilization (%)', alpha=0.5)
    ax1.scatter(nvidia_df['timestamp'], nvidia_df['memory_utilization'], color='purple', marker='.', label='Memory Utilization (%)', alpha=0.5)

    # Customize the first subplot
    ax1.set_ylabel('Metrics')
    ax1.set_title('LLM Performance Metrics Over Time')
    
    # Set y-axis ticks and labels for ax1
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)

    # Add legend for ax1
    ax1.legend(loc='upper left', bbox_to_anchor=(1.15, 1))

    # --- Second Subplot: Prompt Eval TPS ---
    # Plot Prompt Eval TPS (on ax2)
    codestral_prompt_tps = [(ts, tps) for ts, model, tps in prompt_eval_tps if model == 'Codestral']
    mistral_prompt_tps = [(ts, tps) for ts, model, tps in prompt_eval_tps if model == 'Mistral']
    
    if codestral_prompt_tps:
        codestral_prompt_times, codestral_prompt_vals = zip(*codestral_prompt_tps)
        ax2.scatter(codestral_prompt_times, codestral_prompt_vals, color='darkblue', marker='^', label='Codestral Prompt Eval TPS', s=50, alpha=0.7)
    if mistral_prompt_tps:
        mistral_prompt_times, mistral_prompt_vals = zip(*mistral_prompt_tps)
        ax2.scatter(mistral_prompt_times, mistral_prompt_vals, color='darkgreen', marker='^', label='Mistral Prompt Eval TPS', s=50, alpha=0.7)

    # Customize the second subplot
    ax2.set_ylabel('Prompt Eval TPS (tokens/s)')
    
    # Set y-axis range for ax2 (Prompt Eval TPS)
    max_prompt_tps = max([tps for _, _, tps in prompt_eval_tps] + [0])
    if max_prompt_tps == 0:
        ax2.set_ylim(0, 1000)  # Default range: 0 to 1000 tokens/s
    else:
        ax2.set_ylim(0, max_prompt_tps * 1.1)  # Add 10% padding to the top

    # Add legend for ax2
    ax2.legend(loc='upper left', bbox_to_anchor=(1.15, 1))

    # --- Third Subplot: Generation TPS ---
    # Plot Generation TPS (on ax3)
    codestral_gen_tps = [(ts, tps) for ts, model, tps in generation_tps if model == 'Codestral']
    mistral_gen_tps = [(ts, tps) for ts, model, tps in generation_tps if model == 'Mistral']
    
    if codestral_gen_tps:
        codestral_gen_times, codestral_gen_vals = zip(*codestral_gen_tps)
        ax3.scatter(codestral_gen_times, codestral_gen_vals, color='lightblue', marker='v', label='Codestral Generation TPS', s=50, alpha=0.7)
    if mistral_gen_tps:
        mistral_gen_times, mistral_gen_vals = zip(*mistral_gen_tps)
        ax3.scatter(mistral_gen_times, mistral_gen_vals, color='lightgreen', marker='v', label='Mistral Generation TPS', s=50, alpha=0.7)

    # Customize the third subplot
    ax3.set_ylabel('Generation TPS (tokens/s)')
    
    # Set y-axis range for ax3 (Generation TPS)
    max_gen_tps = max([tps for _, _, tps in generation_tps] + [0])
    if max_gen_tps == 0:
        ax3.set_ylim(0, 100)  # Default range: 0 to 100 tokens/s (smaller scale for Generation TPS)
    else:
        ax3.set_ylim(0, max_gen_tps * 1.1)  # Add 10% padding to the top

    # Add legend for ax3
    ax3.legend(loc='upper left', bbox_to_anchor=(1.15, 1))

    # --- Fourth Subplot: TTFT ---
    # Plot TTFT (on ax4)
    codestral_ttft = [(ts, val) for ts, model, val in ttft if model == 'Codestral']
    mistral_ttft = [(ts, val) for ts, model, val in ttft if model == 'Mistral']
    
    if codestral_ttft:
        codestral_ttft_times, codestral_ttft_vals = zip(*codestral_ttft)
        ax4.scatter(codestral_ttft_times, codestral_ttft_vals, color='cyan', marker='s', label='Codestral TTFT (ms)', s=50)
    if mistral_ttft:
        mistral_ttft_times, mistral_ttft_vals = zip(*mistral_ttft)
        ax4.scatter(mistral_ttft_times, mistral_ttft_vals, color='lime', marker='s', label='Mistral TTFT (ms)', s=50)

    # Customize the fourth subplot
    ax4.set_ylabel('TTFT (ms)')
    
    # Set y-axis range for ax4 (TTFT)
    all_ttft = [val for _, _, val in ttft]
    if all_ttft:
        max_ttft = max(all_ttft)
        ax4.set_ylim(0, max_ttft * 1.1)  # Add 10% padding to the top
    else:
        ax4.set_ylim(0, 5000)  # Default range: 0 to 5000 ms (smaller scale for TTFT)

    # Add legend for ax4
    ax4.legend(loc='upper left', bbox_to_anchor=(1.15, 1))

    # --- Fifth Subplot: Total Latency ---
    # Plot Total Response Latency (on ax5)
    codestral_latency = [(ts, val) for ts, model, val in total_latency if model == 'Codestral']
    mistral_latency = [(ts, val) for ts, model, val in total_latency if model == 'Mistral']
    
    if codestral_latency:
        codestral_latency_times, codestral_latency_vals = zip(*codestral_latency)
        ax5.scatter(codestral_latency_times, codestral_latency_vals, color='magenta', marker='d', label='Codestral Total Latency (ms)', s=50)
    if mistral_latency:
        mistral_latency_times, mistral_latency_vals = zip(*mistral_latency)
        ax5.scatter(mistral_latency_times, mistral_latency_vals, color='yellow', marker='d', label='Mistral Total Latency (ms)', s=50)

    # Customize the fifth subplot
    ax5.set_ylabel('Total Latency (ms)')
    ax5.set_xlabel('Time')
    
    # Set y-axis range for ax5 (Total Latency)
    all_total_latency = [val for _, _, val in total_latency]
    if all_total_latency:
        max_total_latency = max(all_total_latency)
        ax5.set_ylim(0, max_total_latency * 1.1)  # Add 10% padding to the top
    else:
        ax5.set_ylim(0, 15000)  # Default range: 0 to 15000 ms

    # Add legend for ax5
    ax5.legend(loc='upper left', bbox_to_anchor=(1.15, 1))

    # Format the x-axis to show time (shared across all subplots)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax5.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'scatter_plot.png'")

    # Close the figure to free memory
    plt.close()

# Main execution
if __name__ == "__main__":
    # Step 1: Find all log files
    main_log_files = glob.glob('log/anything_llm_test.log*') + glob.glob('log/anything_llm_test.log.*')
    container_log_files = glob.glob('log/llama_cpp_metrics_*.log')
    log_files = main_log_files + container_log_files
    if not log_files:
        print("Error: No log files found in log/ directory.")
        exit(1)

    # Step 2: Parse the logs for requests, errors, and performance stats
    client_requests, errors, prompt_eval_tps, generation_tps, ttft, total_latency = parse_anything_llm_logs(log_files)

    # Step 3: Parse the nvidia_smi_metrics.csv file
    nvidia_file = 'log/nvidia_smi_metrics.csv'
    try:
        nvidia_df = parse_nvidia_smi_metrics(nvidia_file)
    except FileNotFoundError:
        print(f"Error: {nvidia_file} not found.")
        exit(1)

    # Step 4: Generate the scatter plots
    generate_scatter_plots(client_requests, errors, prompt_eval_tps, generation_tps, ttft, total_latency, nvidia_df)