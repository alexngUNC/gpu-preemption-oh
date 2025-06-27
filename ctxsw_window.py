import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
import sqlite3
import sys

END2END = 30000
PROC_SPACE = 45
SPACE = 35
TIMESLICE = 2131500

def print_quantum(quantum, index=None):
	start = quantum[1][4]
	end = quantum[0][4]
	if index:
		print(f"===== Quantum {index} =====")
	print(f"Execution window: {[start, end]}, duration: {(end-start)/1e6} ms")
	print(f"           seqNo: {quantum[1][2]}")
	print(f"       contextId: {quantum[0][3]}")

def print_ctxsw_events(ctxsw_events):
	headers = ['tag', 'PID', 'vmID', 'seqNo', 'contextId', 'timestamp', 'gpuId']
	table = tabulate(ctxsw_events, headers=headers, tablefmt='grid')
	print(table)

def accumulate_ctx_exec_time(ctxsw_events, first_quantum_start, last_quantum_end):
	"""
	Accumulate execution time of a particular GPU context (same PID and contextId)
	"""
	assert(len(ctxsw_events) > 0)
	assert(len(ctxsw_events) % 2 == 0)
	assert(ctxsw_events[0][0]==8)
	last_tag = None
	last_globalPid = None
	last_contextId = None
	last_timestamp = None
	total_runtime = 0
	quanta_starts = 0
	for evt in ctxsw_events:
		curr_tag = evt[0]
		curr_globalPid = evt[5]
		curr_contextId = evt[3]
		curr_timestamp = evt[4]
		if curr_tag == 7:
			if (last_tag == 8):
				duration = curr_timestamp - last_timestamp
				total_runtime += duration
			else:
				print("Warning: dropping a ctxsw event")
		elif curr_tag == 8:
			assert(last_tag == 7 or last_tag == None)
			quanta_starts += 1
		last_timestamp = curr_timestamp
		last_tag = curr_tag
		last_contextId = curr_contextId
		last_globalPid = curr_globalPid
	last_ctxsw_event = ctxsw_events[-1]
	assert(last_ctxsw_event[0]==7)
	return total_runtime


def get_ctxsw_overhead(ctxsw_events):
	num_events = len(ctxsw_events)
	assert(num_events >= 4)
	assert(num_events % 2 == 0)
	assert(ctxsw_events[0][0] == 8 and ctxsw_events[-1][0] == 7)
	save_end = 1
	restore_start = 2
	all_overheads = []
	count = 0
	total_overhead = 0
	while restore_start < num_events-1:
		save_end_evt = ctxsw_events[save_end]
		restore_start_evt = ctxsw_events[restore_start]
		assert(save_end_evt[0] == 7 and restore_start_evt[0] == 8)
		overhead = restore_start_evt[4] - save_end_evt[4]
		total_overhead += overhead
		all_overheads.append(overhead)
		count += 1
		save_end += 2
		restore_start += 2
	assert(count == (num_events/2)-1)
	return total_overhead, count, all_overheads

def get_real_ctxsw_overhead(ctxsw_events):
	"""
	Intended usage is to input all context switch events
	"""
	num_events = len(ctxsw_events)
	assert(num_events >= 4)
	assert(num_events % 2 == 0)
	assert(ctxsw_events[0][0] == 8 and ctxsw_events[-1][0] == 7)
	quantum_start = 0
	save_end = 1
	restore_start = 2
	all_overheads = []
	count = 0
	total_overhead = 0
	while restore_start < num_events-1:
		save_end_evt = ctxsw_events[save_end]
		restore_start_evt = ctxsw_events[restore_start]
		assert(save_end_evt[0] == 7 and restore_start_evt[0] == 8)
		overhead = restore_start_evt[4] - save_end_evt[4]
		quantum_start_evt = ctxsw_events[quantum_start]
		assert(quantum_start_evt[0] == 8)
		exec_time = save_end_evt[4] - quantum_start_evt[4]
		if exec_time > TIMESLICE:
			overhead += exec_time - TIMESLICE
		total_overhead += overhead
		all_overheads.append(overhead)
		count += 1
		save_end += 2
		restore_start += 2
		quantum_start += 2
	assert(count == (num_events/2)-1)
	return total_overhead, count, all_overheads


def get_real_ctxsw_overhead_stats(ctxsw_events):
	"""
	Input: all context switch events
	Returns: stats regarding the context switch overhead of
	each context
	"""
	num_events = len(ctxsw_events)
	assert(num_events >= 4)
	assert(num_events % 2 == 0)
	assert(ctxsw_events[0][0] == 8 and ctxsw_events[-1][0] == 7)
	quantum_start = 0
	save_end = 1
	restore_start = 2
	# [{'Context ID': 0, 'Mean': 0, 'Min': 0, 'Q1': 0, 'Median': 0, 'Q3': 0, 'Max': 0}]
	ctxsw_stats = []
	ctx2oh = {} # ctxid: [40us, 40us, 50us, ...]
	while restore_start < num_events-1:
		save_end_evt = ctxsw_events[save_end]
		restore_start_evt = ctxsw_events[restore_start]
		assert(save_end_evt[0] == 7 and restore_start_evt[0] == 8)
		overhead = restore_start_evt[4] - save_end_evt[4]
		quantum_start_evt = ctxsw_events[quantum_start]
		assert(quantum_start_evt[0] == 8)
		exec_time = save_end_evt[4] - quantum_start_evt[4]
		if exec_time > TIMESLICE:
			overhead += exec_time - TIMESLICE
		ctxid = quantum_start_evt[3]
		assert(ctxid == save_end_evt[3])
		if ctxid not in ctx2oh:
			ctx2oh[ctxid] = []
		ctx2oh[ctxid].append(overhead)
		save_end += 2
		restore_start += 2
		quantum_start += 2
	for ctx, ohs in ctx2oh.items():
		stats = {}
		cur_ohs = np.array(ohs)
		stats['Context ID'] = ctx
		stats['Mean'] = np.mean(cur_ohs)
		stats['Min'] = np.min(cur_ohs)
		stats['Q1'] = np.percentile(cur_ohs, 25)
		stats['Median'] = np.median(cur_ohs)
		stats['Q3'] = np.percentile(cur_ohs, 75)
		stats['Max'] = np.max(cur_ohs)
		ctxsw_stats.append(stats)
	return ctxsw_stats

def get_exec_time_stats(ctxsw_events):
	"""
	Assume that the input events are under one PID and CTX ID
	"""
	num_events = len(ctxsw_events)
	exec_stats = {'Context ID': 0, f'Mean': 0, f'Min': 0, 
		f'Q1': 0, f'Median': 0, f'Q3': 0, f'Max': 0, 'Num Quanta': 0}
	assert(num_events >= 2)
	assert(num_events % 2 == 0)
	assert(ctxsw_events[0][0] == 8 and ctxsw_events[-1][0] == 7)
	start = 0
	end = 1
	total_exec_time = 0
	cur_pid = ctxsw_events[0][5]
	cur_ctxid = ctxsw_events[0][3]
	exec_stats['Context ID'] = cur_ctxid
	all_exec_blocks = []
	while end < num_events:
		start_evt = ctxsw_events[start]
		end_evt = ctxsw_events[end]
		assert(start_evt[5] == cur_pid and end_evt[5] == cur_pid)
		assert(start_evt[3] == cur_ctxid and end_evt[3] == cur_ctxid)
		assert(start_evt[0] == 8 and end_evt[0] == 7)
		exec_time = end_evt[4] - start_evt[4]
		total_exec_time += exec_time
		all_exec_blocks.append(exec_time)
		start += 2
		end += 2
	blocks = np.array(all_exec_blocks)
	exec_stats[f'Mean'] = np.mean(blocks)
	exec_stats[f'Min'] = np.min(blocks)
	exec_stats[f'Q1'] = np.percentile(blocks, 25)
	exec_stats[f'Median'] = np.median(blocks)
	exec_stats[f'Q3'] = np.percentile(blocks, 75)
	exec_stats[f'Max'] = np.max(blocks)
	exec_stats['Num Quanta'] = len(blocks)
	exec_stats[f'Total Time'] = total_exec_time
	assert(exec_stats['Num Quanta'] == (num_events / 2))
	return exec_stats


if __name__ == "__main__":
	if len(sys.argv) < 4:
		raise ValueError("Please provide a sqlite db file and analysis window start and end times.")
	parser = argparse.ArgumentParser(description='Summarize GPU execution time on per-process and per-context bases')
	parser.add_argument('db_file', type=str, help='Path to the database file (e.g., vit-huge-inter-1-eevdf.db)')
	parser.add_argument('window_start', type=int, help='Start timestamp of analysis window in ns')
	parser.add_argument('window_end', type=int, help='End timestamp of analysis window in ns')
	parser.add_argument('--units', type=str, default='ms', help='Time units')
	args = parser.parse_args()

	# analysis window start and end times
	window_start = args.window_start
	window_end = args.window_end
	units = args.units

	# units of time to display
	div = 1
	if args.units == "us":
		div = 1e3
	elif args.units == "ms":
		div = 1e6
	elif args.units == "s":
		div = 1e9
	else:
		units = "ns"

	# setup sqlite connection
	db_file = args.db_file
	conn = sqlite3.connect(db_file)
	cursor = conn.cursor()

	# Get start of first quantum for our program
	query = """
	SELECT * FROM GPU_CONTEXT_SWITCH_EVENTS 
	WHERE timestamp < ? 
	ORDER BY timestamp DESC 
	LIMIT 2;
	"""
	cursor.execute(query, [window_start])
	first_quantum = cursor.fetchall()
	assert(len(first_quantum)==2 and first_quantum[0][0]==7 and first_quantum[1][0]==8)
	assert(first_quantum[0][3] == first_quantum[1][3])
	first_quantum_start = first_quantum[1][4]
	print("========== First Quantum ==========")
	print_quantum(first_quantum)

	# get last quantum
	query = """
	SELECT * FROM GPU_CONTEXT_SWITCH_EVENTS
	WHERE timestamp < ?
	ORDER BY timestamp DESC
	LIMIT 2;
	"""
	cursor.execute(query, [window_end])
	last_quantum = cursor.fetchall()
	assert(len(last_quantum)==2 and last_quantum[0][0]==7 and last_quantum[1][0]==8)
	assert(last_quantum[0][3] == last_quantum[1][3])
	print("========== Last Quantum ==========")
	print_quantum(last_quantum)
	last_quantum_end = last_quantum[0][4]

	# end2end analysis window: [first quantum start - last quantum end]
	end2end_time = last_quantum_end - first_quantum_start
	print(f"End2end duration: {end2end_time} ns = {end2end_time/1e6} ms = {end2end_time/1e9} s")

	# get PID and contextId mappings
	query = """
	SELECT globalPid, GROUP_CONCAT(DISTINCT contextId) AS contextIds
	FROM GPU_CONTEXT_SWITCH_EVENTS
	WHERE timestamp >= ?
	AND timestamp <= ?
	GROUP BY globalPid;
	"""
	cursor.execute(query, [first_quantum_start, last_quantum_end])
	r = cursor.fetchall()
	# PID: [contextId0, contextId1,...]
	pid2ctx = {}
	for globalPid, contextIds in r:
		pid2ctx[globalPid] = [int(x) for x in contextIds.split(',')]
	
	# sanity check that first and last quantum are the same as the ones fetched before
	query = """
	SELECT * FROM GPU_CONTEXT_SWITCH_EVENTS
	WHERE timestamp >= ?
	AND timestamp <= ?
	AND (tag = 8 OR tag = 7)
	ORDER BY timestamp ASC;
	"""
	cursor.execute(query, [first_quantum_start, last_quantum_end])
	all_contexts = cursor.fetchall()
	first_quantum_analyzed = []
	first_quantum_analyzed.append(all_contexts[1])
	first_quantum_analyzed.append(all_contexts[0])
	assert(first_quantum_analyzed[0][3] == first_quantum_analyzed[1][3])
	assert(first_quantum_analyzed[0][2] == first_quantum[0][2])
	last_quantum_analyzed = []
	last_quantum_analyzed.append(all_contexts[-1])
	last_quantum_analyzed.append(all_contexts[-2])
	assert(last_quantum_analyzed[0][3] == last_quantum_analyzed[1][3])
	assert(last_quantum_analyzed[0][2] == last_quantum[0][2])
	#print("========== First Quantum Analyzed ==========")
	#print_quantum(first_quantum_analyzed)
	#print("========== Last Quantum Analyzed ==========")
	#print_quantum(last_quantum_analyzed)

	# get execution time for each context
	ctx2time = {} # [contextId: 2003 ms] (one GPU context)
	proc2time = {} # [PID: 201230 ms] (all GPU contexts under PID)
	ctx_exec_stats = [] # [{CTXID 1 stats}]
	for pid, ctxs in pid2ctx.items():
		proc_time = 0
		for ctx in ctxs:
			query = """
			SELECT * FROM GPU_CONTEXT_SWITCH_EVENTS
			WHERE globalPid == ?
			AND contextId == ?
			AND timestamp >= ?
			AND timestamp <= ?
			AND (tag = 8 OR tag = 7)
			ORDER BY timestamp ASC;
			"""
			cursor.execute(query, [pid, ctx, first_quantum_start, last_quantum_end])
			ctxsw_events = cursor.fetchall()
			# TODO: are ctx ids unique across different processes?
			curctx_stats = get_exec_time_stats(ctxsw_events)
			ctx_exec_stats.append(curctx_stats)
			ctx_time = accumulate_ctx_exec_time(ctxsw_events, first_quantum_start, last_quantum_end)
			ctx2time[ctx] = ctx_time
			proc_time += ctx_time
		proc2time[pid] = proc_time

	# Initialize lists for the table
	context_ids = []
	pids = []
	proc_times = []
	ctx_times = []

	# Iterate over the dictionary to group context IDs by PID
	for pid, context_list in pid2ctx.items():
		 # Split context IDs and associate them with the current PID
		context_ids.extend(context_list)
		numctxs = len(context_list)
		pids.extend([pid]*numctxs)
		curproc = 0
		for ctx in context_list:
			curproc += ctx2time[ctx]
			ctx_times.append(ctx2time[ctx])
		proc_times.extend([curproc]*numctxs)
	# create dataframe for table
	assert len(ctx_times) == len(context_ids)
	df = pd.DataFrame({
		'PID': pids,
		'Process-wide GPU Execution Time': proc_times,
		'GPU Context ID': context_ids,
		'Context Execution Time': ctx_times
	})

	# Initialize a list to store the rows with separators
	final_rows = []

	# We need to track when the PID changes
	current_pid = None

	# Loop through the rows of the DataFrame
	last_pid = df.iloc[-1]['PID']
	for i, row in df.iterrows():
		if row['PID'] != current_pid:
			#if current_pid is not None:
				#final_rows.append(['-'*SPACE, '-'*PROC_SPACE, '-'*SPACE, '-'*SPACE])
			# Add the PID row (only if it's a new PID)
			final_rows.append([row['PID'], row['Process-wide GPU Execution Time']/div, row['GPU Context ID'], row['Context Execution Time']/div])
			#Add a line separator after the PID row (line spans across both columns)
			current_pid = row['PID']
		else:
			# Add context ID with empty PID for subsequent context IDs of the same PID
			final_rows.append(['', '', '-'*SPACE, '-'*SPACE])
			final_rows.append(['', '', row['GPU Context ID'], row['Context Execution Time']/div])
		if current_pid != last_pid:
			final_rows.append(['-'*SPACE, '-'*PROC_SPACE, '-'*SPACE, '-'*SPACE])
	# Convert the list back to a DataFrame
	final_df = pd.DataFrame(final_rows, columns=['PID', f'Process-wide GPU Execution Time ({units})', 'GPU Context ID', f'Context Execution Time ({units})'])

	# Print the table using tabulate
	print("-"*163)
	print(tabulate(final_df, headers='keys', tablefmt='pipe', showindex=False))
	print("-"*163)
	overhead = (end2end_time/div)-(sum(proc2time.values())/div)
	print(f"Total context switch overhead: {overhead} {units}")

	# get all context switch events in the window across all processes
	query = """
	SELECT * FROM GPU_CONTEXT_SWITCH_EVENTS
	WHERE timestamp >= ?
	AND timestamp <= ?
	AND (tag = 8 or tag = 7)
	ORDER BY timestamp ASC;
	"""
	cursor.execute(query, [first_quantum_start, last_quantum_end])
	ctxsw_events = cursor.fetchall()
	total_overhead, num_ctxsw, all_overheads = get_real_ctxsw_overhead(ctxsw_events)
	ctxsw_overheads = np.array(all_overheads)
	min_ctxsw_oh = np.min(ctxsw_overheads)
	max_ctxsw_oh = np.max(ctxsw_overheads)
	median_ctxsw_oh = np.median(ctxsw_overheads)
	mean_ctxsw_oh = np.mean(ctxsw_overheads)
	q1_ctxsw_oh = np.percentile(ctxsw_overheads, 25)
	q3_ctxsw_oh = np.percentile(ctxsw_overheads, 75)
	print("\n"+'='*6+' All Context Switch Overheads Summary '+'='*7)
	print(f"Total context switch overhead: {total_overhead/div} {units}")
	print(f"Number of context switches: {num_ctxsw}")
	print(f"Min context switch overheads: {min_ctxsw_oh/div} {units}")
	print(f"25th percentile context switch overhead: {q1_ctxsw_oh/div} {units}")
	print(f"Median context switch overheads: {median_ctxsw_oh/div} {units}")
	print(f"75th percentile context switch overheads: {q3_ctxsw_oh/div} {units}")
	print(f"Max context switch overhead: {max_ctxsw_oh/div} {units}")
	print(f"Mean context switch overhead: {mean_ctxsw_oh/div} {units}")
	print('='*51+"\n")

	# Context switch overhead data per context
	query = """
	SELECT * FROM GPU_CONTEXT_SWITCH_EVENTS
	WHERE timestamp >= ?
	AND timestamp <= ?
	AND (tag = 8 or tag = 7)
	ORDER BY timestamp ASC;
	"""
	cursor.execute(query, [first_quantum_start, last_quantum_end])
	ctxsw_events = cursor.fetchall()
	ctxsw_stats = get_real_ctxsw_overhead_stats(ctxsw_events)
	df = pd.DataFrame(ctxsw_stats)
	df[['Mean', 'Min', 'Q1', 'Median', 'Q3', 'Max']] = df[['Mean', 'Min', 'Q1', 'Median', 'Q3', 'Max']] / div
	print('\n'+'='*16 + f' Context Switch Overhead Per Context ({units}) ' + '='*17)
	print('-'*75)
	print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
	print('-'*75+'\n')

	# Exec times stats
	df = pd.DataFrame(ctx_exec_stats)
	df[[f'Mean', f'Min', f'Q1', f'Median', 
			f'Q3', f'Max', f'Total Time']] = df[[f'Mean', f'Min', f'Q1', f'Median', 
			f'Q3', f'Max', f'Total Time']] / div
	print('='*39 + f' Context Execution Time ({units}) ' + '='*39)
	print("-"*107)
	print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
	print("-"*107)
