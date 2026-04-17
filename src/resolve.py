"""Handling multi-threading, GPU, input parsing, writing"""

import mrcfile
import numpy as np
import os
import multiprocessing as mp
import datetime
import gc
from . import core, utils, utils_misc
import torch
from dataclasses import dataclass
from copy import copy

@dataclass
class ProcessingJob:
	"""Configuration for a single half-map processing run.

	Bundles all parameters needed by :func:`process_one_file` into a
	single object, replacing the positional tuple pattern required by
	``multiprocessing.Pool.map``.

	Parameters
	----------
	mode : str
		Processing mode. ``'single'`` or ``'batch'``.
	config : dict
		Configuration dictionary containing algorithm parameters.
	apix : float
		Voxel size in Ångströms per pixel (Å px⁻¹).
	odd_input : str
		File path to the first (odd) half-map.
	even_input : str
		File path to the second (even) half-map.
	cpu_threads : int
		Number of CPU threads available for computation.
	gpu_enabled : bool
		Whether to use GPU acceleration.
	gpu_settings : list[int] or None
		List of GPU device IDs. Currently only the first is used.
	run_fast : bool
		If ``True``, skip intermediate resolution shells and use a
		larger step size.
	mask_strategy : str
		Masking strategy. ``'remove_background'``, ``'signal_mask'``,
		or ``'full_map'``.
	signal_mask_input : str or None
		File path to a binary mask. Used when *mask_strategy* is
		``'signal_mask'``.
	mask_measure : str
		Summary statistic for global resolution. ``'average'`` or
		``'median'``.
	outputDir : str
		Directory path where result files will be written.
	test2 : bool
		Whether to apply a secondary statistical test.
	p_cutoff : float
		P-value cutoff for significance testing.
	resMax : float
		Maximum resolution in Ångströms to evaluate.
	accuracy_steps : int
		Number of interpolation steps for sub-shell resolution
		accuracy.
	referenceDistSize : int
		Number of correlation samples for reference distribution
		construction.
	spacingFilter : float
		Width of the bandpass filter relative to Nyquist.
	falloff : float
		Sharpness of the bandpass filter roll-off.
	"""

	mode: str
	config: dict
	apix: float | None
	odd_input: str
	even_input: str
	cpu_threads: int
	gpu_enabled: bool
	gpu_settings: int | None
	run_fast: bool
	mask_strategy: str
	signal_mask_input: str | None
	mask_measure: str
	outputDir: str
	test2: bool
	p_cutoff: float
	resMax: float
	accuracy_steps: int
	referenceDistSize: int
	spacingFilter: float
	falloff: float

def process_one_file(job: ProcessingJob) -> tuple[str, float, float] | None:
	"""Process a single half-map pair in a worker process.

	Parameters
	----------
	job : ProcessingJob
		All parameters for this processing run.

	Returns
	-------
	tuple[str, float, float] or None
		``(name, resolution, signal_ratio)``.
	"""

	# Configurations for running on GPU
	gpu_id = None
	if job.gpu_enabled:
		gpu_id = job.gpu_settings
  
	if torch.cuda.is_available() and gpu_id is not None:
		device = torch.device(f"cuda:{gpu_id}")
		print(f'GPU processing: {gpu_id}')
	else:
		device = torch.device("cpu")
		print("CPU processing")

	# Safety checks
	if not job.odd_input.endswith((".mrc", ".map")):
		return None
	if job.odd_input == job.even_input:
		print("Error: given same path twice. End.")
		return None

	# Naming file
	preAddToName = job.odd_input.split("/")[-1][:-4] + "_" + job.config + "_locRes"  
	outputFilename_LocRes = os.path.join(job.outputDir, preAddToName + ".mrc")

	# Initializations and reading data
	halfMap1 = mrcfile.open(job.odd_input, mode='r')
	halfMap2 = mrcfile.open(job.even_input, mode='r')
	if job.mode != "batch":
		print("\nusing input half-maps: ")
		print(job.even_input)
		print(job.odd_input)
		print("")
  
  
	halfMap1Data = halfMap1.data
	halfMap2Data = halfMap2.data
	sizeMap = halfMap1Data.shape
	dimension = len(sizeMap)

	# More safety checks
	if job.config == "Refined-Maps":
		if dimension != 3:
			print("Error: inputs should be 3D")
			return None
	if job.config == "Micrographs":
		if dimension != 2:
			print("Error: inputs should be 2D")
			return None    
	if job.config == "Tilt-Series":
		if dimension != 3:
			print("Error: inputs should be 3D")
			return None    
	if job.config == "Tomograms":
		if dimension != 3:
			print("Error: inputs should be 3D")
			return None
	if halfMap1Data.shape != halfMap2Data.shape:
		print("input maps do not have same size. Exit.")
		return None

	# Configuring step size dependent on input data dimensions
	if dimension == 2:
		stepSize = [5,5]
	else:
		stepSize = [2,2,2]
		if job.run_fast:
			stepSize = [3,3,3]

	# Processing signal mask for median estimate
	signal_mask = None
	if job.mask_strategy == "signal_mask":
		if not os.path.exists(job.signal_mask_input):
			print("Signal mask not found. Exit.")
			return None
		if len(job.signal_mask_input) == 0:
			signal_mask = None
		else:
			# signal_mask = torch.from_numpy(mrcfile.open(signal_mask_input).data * 1).float().to(device)
			signal_mask = mrcfile.open(job.signal_mask_input).data * 1
			if signal_mask.shape != halfMap1Data.shape:
				print("Signal mask has not shape of input map. Exit.")
				return None
			signal_mask[signal_mask < 1] = 0
			signal_mask[signal_mask >= 1] = 1
			print("using signal mask for median estimate: " + str(job.signal_mask_input))
	if job.mask_strategy == "full_map":
		# signal_mask = torch.ones(halfMap1Data.shape).to(device)
		signal_mask = np.ones(halfMap1Data.shape)

	# Reading pixel size
	if job.mode != "batch": print("Input configurations_____________")
	if job.apix is None:
		apix = np.round(float((halfMap1.voxel_size).x),2)
		apix_y = np.round(float((halfMap1.voxel_size).y),2)
		if dimension == 2:
			if job.mode != "batch": print("pixel size read from header (x,y): " + str(apix) + " " + str(apix_y))
		if dimension == 3:
			apix_z = np.round(float((halfMap1.voxel_size).z),2)
			if job.mode != "batch": print("pixel size read from header (x,y,z): " + str(apix) + " " + str(apix_y) + " " + str(apix_z)) # Z-value may differ for Tilt-series
	else:
		apix = float(job.apix)
	lowRes = job.resMax*apix # Lowest resolution 
	lowResMax = 1/(np.fft.rfftfreq(np.min(sizeMap))[1]/apix)
	lowRes = np.min([lowRes, lowResMax])
	if job.mode != "batch": print("lowest resolution to consider (10*apix): " + str(np.round(lowRes,2)))

	# This is for tilt-series 
	if job.config == "Tilt-Series":
		dimension_windows = 2
		stepSize = [5,5,5]
		stepSize[0] = 1 # Collapse z to 1. Not that for numpy arrays, x and z are swapped (z,y,x instead of x,y,z)
	else:
		dimension_windows = dimension
	if job.mode != "batch": print("using step size: " + str(" ".join(np.array(stepSize[::-1]).astype(str)))) # Adjust x-z swap

	# Get windows (radii) and shells
	sizeVol = 100
	resolutions = utils_misc.calculate_resolutions(sizeVol, apix, lowRes, job.accuracy_steps)

	# Get windows and box sizes from precalculated empirical simulations. Print out parameters used.
	windows = utils_misc.get_windows_empirical(apix, resolutions, dimension_windows)
	if job.mode != "batch": 
		print("using window radii [pix]: " + " ".join(map(str,np.round(windows,1))))
		print("to measure resolutions [Å]: " + " ".join(map(str,np.round(resolutions,2))))
		print("")

	# Preparation as batches
	if job.config == "Tilt-Series":
		batchHalf1 = [torch.from_numpy(halfMap1Data[i].astype(np.float32)).to(device) for i in range(halfMap1Data.shape[0])]
		batchHalf2 = [torch.from_numpy(halfMap2Data[i].astype(np.float32)).to(device) for i in range(halfMap2Data.shape[0])]
	else:
		batchHalf1 = [torch.from_numpy(halfMap1Data.astype(np.float32)).to(device)]
		batchHalf2 = [torch.from_numpy(halfMap2Data.astype(np.float32)).to(device)]


	phasePermutation = True
	if job.config == "Refined-Maps":
		phasePermutation = False

	# Use batch processing here only for Tilt-series! 
	locResMap = core.compute_resolution(
		apix,
		windows,
		resolutions,
		batchHalf1,
		batchHalf2,
		stepSize[-1],
		gpu_id,
		job.referenceDistSize,
		phasePermutation,
		4096,
		job.spacingFilter,
		job.falloff)
 
	if job.config != "Tilt-Series":
		locResMap = locResMap[0]
	else:
		locResMap = locResMap.permute(1, 0, 2, 3)

	# Naming file
	preAddToName = job.odd_input.split("/")[-1][:-4] + "_" + job.config + "_locRes"  
	outputFilename_LocRes = os.path.join(job.outputDir, preAddToName + ".mrc")
	pValueMapShape = locResMap[0].shape

	# cleanup 1
	del batchHalf1, batchHalf2
	torch.cuda.empty_cache()
	gc.collect()


	localResMap_out = utils_misc.fill_map(locResMap, torch.tensor(resolutions, device=device, dtype = locResMap[0].dtype), job.p_cutoff, lowRes, job.test2)
	localResMap_out[localResMap_out>lowRes] = lowRes   

	# cleanup 2
	locResMap = locResMap.cpu().numpy()
	torch.cuda.empty_cache()

	# median p-value creation
	actualRes_global_new, signalRatio = None, None
	if job.config != "Refined-Maps":
		if signal_mask is None:
			# signalMask_stepSize = torch.ones(pValueMapShape, device=device)
			signalMask_stepSize = np.ones(pValueMapShape)
			signalMask_stepSize[localResMap_out.cpu().numpy() >= lowRes] = 0
		else:
			if dimension == 2:
				signalMask_stepSize = signal_mask[::stepSize[0], ::stepSize[1]]   
			if dimension == 3:
				signalMask_stepSize = signal_mask[::stepSize[0], ::stepSize[1], ::stepSize[2]]   				
		dict_tilt_series, signalRatio = utils_misc.calculate_median_res(locResMap, signalMask_stepSize, resolutions, dimension, job.config, job.mask_measure)
		actualRes_global_new = utils_misc.write_medianRes(dict_tilt_series, signalRatio, resolutions, job.p_cutoff, lowRes, job.config, job.mode, job.outputDir, preAddToName, job.mask_measure)


	# cleanup 3
	del locResMap
	if signal_mask is not None:
		del signal_mask
	if job.config != "Refined-Maps":
		del signalMask_stepSize
	torch.cuda.empty_cache()
	gc.collect()

	# The grid needs to be interpolated in the end.
	print("interpolating grid")

	if np.max(stepSize) != 1: # This is always the case in this default script
		if job.config == "Tilt-Series": # For tilt-series
			localResMap = []
			for i in range(localResMap_out.shape[0]):
				localResMap.append(utils_misc.interpolate_with_zoom(localResMap_out[i], sizeMap[1:], stepSize, lowRes))
			localResMap = torch.stack(localResMap)
		else:
			localResMap = utils_misc.interpolate_with_zoom(localResMap_out, sizeMap, stepSize, lowRes)
	else:
		localResMap = np.copy(localResMap_out)    
	# del localResMap_out 
	localResMap[torch.isnan(localResMap)] = lowRes

	localResMap = localResMap.cpu().numpy()


	# Write output
	print("write here: " + str(outputFilename_LocRes))
	localResMapMRC = mrcfile.new(outputFilename_LocRes, overwrite=True)
	localResMap = np.float32(localResMap)
	localResMapMRC.set_data(localResMap)
	if job.config == "Tilt-Series":
		localResMapMRC.voxel_size = (apix,apix,apix)
	else:
		localResMapMRC.voxel_size = apix
	localResMapMRC.close()
	

	# For 2D calculations (micrographs), also save 2D image as output.
	if dimension==2:
		import matplotlib.pyplot as plt
		localResMap = np.flipud(localResMap) # Correct axis swap for coherent visualization.
		plt.rcParams['font.size'] = 16
		cmap = plt.get_cmap('bwr')
		fig, ax = plt.subplots() 
		img = ax.imshow(localResMap, cmap=cmap, vmin=2*apix, vmax=lowRes)
		cbar = plt.colorbar(img, ax=ax, pad=0.05, aspect=20)
		cbar.set_label('Resolution') 
		cbar.ax.invert_yaxis()
		num_ticks = 6
		cbar_ticks = np.linspace(2*apix, lowRes, num_ticks)
		cbar.set_ticks(cbar_ticks)
		cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks])
		plt.tight_layout()
		plt.axis('off')
		plt.savefig(outputFilename_LocRes[:-3]+"png", bbox_inches='tight', pad_inches=0.05) 

	return preAddToName, actualRes_global_new, signalRatio
 


def main(
	mode: str,
	config: dict,
	apix: float | None,
	odd_input: str | list[str],
	even_input: str | list[str],
	cpu_threads: int,
	gpu_enabled: bool,
	gpu_settings: list[int] | None,
	run_fast: bool,
	mask_strategy: str,
	signal_mask_input: str | None,
	mask_measure: str,
	outputDir: str,
	inputDir: str,
) -> None:
	"""Run the local resolution estimation pipeline.

	Estimates local resolution from two independent half-maps by measuring
	local Fourier shell correlations at multiple resolution shells. Supports
	single-map and batch processing modes.

	Parameters
	----------
	mode : str
		Processing mode. ``'single'`` processes one pair of half-maps;
		``'batch'`` processes multiple pairs in sequence.
	config : dict
		Configuration dictionary containing algorithm parameters such as
		resolution shells, step size, bandpass filter settings, and
		statistical thresholds.
	apix : float
		Voxel size in Ångströms per pixel (Å px⁻¹).
	odd_input : str or list[str]
		File path(s) to the first (odd) half-map(s). A single path in
		``'single'`` mode or a list of paths in ``'batch'`` mode.
	even_input : str or list[str]
		File path(s) to the second (even) half-map(s). Must match the
		length and element shapes of *odd_input*.
	cpu_threads : int
		Number of CPU threads available for computation.
	gpu_enabled : bool
		Whether to use GPU acceleration. If ``False``, all computation
		is performed on the CPU.
	gpu_settings : list[int] or None
		GPU device ID to use when *gpu_enabled* is ``True``.
		 ``None`` when GPU is disabled.
	run_fast : bool
		If ``True``, accelerate computation by skipping intermediate
		resolution shells and using a larger step size, reducing spatial
		and Fourier-space sampling density. Trades precision for speed.
	mask_strategy : str
		Strategy for masking the map during global resolution estimation.
		``'remove_background'`` automatically removes regions that do not
		pass the lowest measured resolution. ``'signal_mask'`` applies a
		user-provided binary mask given by *signal_mask_input*.
		``'full_map'`` uses the entire map without masking.
	signal_mask_input : str or None
		File path to a binary mask defining the region of interest. Only
		used when *mask_strategy* is ``'signal_mask'``; ignored otherwise.
	mask_measure : str
		Summary statistic used for global resolution determination.
		``'average'`` computes the mean; ``'median'`` computes the median
		over masked voxels.
	outputDir : str
		Directory path where result files will be written.
	inputDir : str
		Directory path containing the input half-map files.

	Returns
	-------
	None
		Results are written to *outputDir*.
	"""

	if not os.path.exists(outputDir):
		os.makedirs(outputDir)

	start_total = datetime.datetime.now()

	# Some configurations for the different input types
	if config == "Refined-Maps":
		test2 = False	
		p_cutoff = 0.01
	if config == "Micrographs":
		test2 = True
		p_cutoff = 0.05
	if config == "Tilt-Series":
		test2 = True
		p_cutoff = 0.05
	if config == "Tomograms":
		test2 = True
		p_cutoff = 0.05

	# Inputs
	resMax = 10
	accuracy_steps = 1
	referenceDistSize = 10000
	spacingFilter = 0.05
	falloff = 1.5
	if run_fast:
		accuracy_steps = 2


   
	# Parse input GPUs			
	inputGPUs = None
	if gpu_settings != "Disabled":
		if len(gpu_settings) != 0: inputGPUs = list(np.array(gpu_settings.split(",")).astype(int))
		if len(gpu_settings) == 0: inputGPUs = list(range(torch.cuda.device_count()))

	# Job that gets passed to process_one_file
	shared_job = ProcessingJob(
		mode=mode,
		config=config,
		apix=apix,
		odd_input=None,      # filled in per file pair later
		even_input=None,      # filled in per file pair later
		cpu_threads=cpu_threads,
		gpu_enabled=gpu_enabled,
		gpu_settings=inputGPUs,
		run_fast=run_fast,
		mask_strategy=mask_strategy,
		signal_mask_input=signal_mask_input,
		mask_measure=mask_measure,
		outputDir=outputDir,
		test2=test2,
		p_cutoff=p_cutoff,
		resMax=resMax,
		accuracy_steps=accuracy_steps,
		referenceDistSize=referenceDistSize,
		spacingFilter=spacingFilter,
		falloff=falloff,
	)

	# Handle single mode processing
	if mode != "batch":
		if inputGPUs is not None: # for single mode processing, only one GPU can be used.
			inputGPUs = inputGPUs[0]
			shared_job.gpu_settings = inputGPUs
		outputFilename_LocRes = os.path.join(outputDir, os.path.basename(odd_input)[:-4] + "_" + config + "_locRes" + ".mrc")
		if os.path.exists(outputFilename_LocRes):
			print("Warning: " + outputFilename_LocRes + " already exists. This file is processed already. For reprocessing, please delete output file or define new output directory. SKIP!\n")
			return     
		shared_job.odd_input = odd_input   # odd_input
		shared_job.even_input = even_input   # even_input
		process_one_file(shared_job)
		print("IN TOTAL: " + str(datetime.datetime.now()-start_total) + "\n\n")
		return

	# ____________Batch mode__________
 
	
	# handle multi-threading
	if gpu_enabled:
		global_threats = 1
		cpuThreads_batch = cpu_threads
		if len(inputGPUs) > 1:
			global_threats = len(inputGPUs)
			cpuThreads_batch = np.max([1, int(cpuThreads_batch//len(inputGPUs))])
	else:
		minThreads = 1
		global_threats = 1
		if cpu_threads > minThreads:
			global_threats = int(cpu_threads//minThreads)
			cpuThreads_batch = minThreads
 
	# handle input
	odd_id = odd_input
	even_id = even_input
	matching_files = [f for f in os.listdir(inputDir) if odd_id in f]
	it_loops = len(matching_files)

	# Build list of args for each file
	all_jobs = []
	for iterate_files in range(it_loops):
		odd_file = matching_files[iterate_files]
		even_file = odd_file.replace(odd_id, even_id)
		odd_path = os.path.join(inputDir, odd_file)

		outputFilename_LocRes = os.path.join(outputDir, os.path.basename(odd_path)[:-4] + "_" + config + "_locRes" + ".mrc")
		if os.path.exists(outputFilename_LocRes):
			print("Warning: " + outputFilename_LocRes + " already exists. This file is processed already. For reprocessing, please delete output file or define new output directory. SKIP!\n")
			continue

		even_path = os.path.join(inputDir, even_file)
		job = copy(shared_job)
		job.odd_input = odd_path   # odd_input
		job.even_input = even_path   # even_input
		job.cpu_threads = cpuThreads_batch
		if global_threats > 1:
			if (gpu_settings != "Disabled") and (global_threats != 0) and (inputGPUs is not None):
				job.gpu_settings = inputGPUs[iterate_files%global_threats]
		else:
			if inputGPUs is not None: 
				job.gpu_settings = inputGPUs[0]
				print(inputGPUs)
				

		all_jobs.append(job)

	nFiles = len(all_jobs)
	print(f"Batch mode: found {nFiles} files to process")



	# Process files, each in a fresh worker
	nameArray, resGlobArray, ratioSignalArray = [], [], []
	result_queue = mp.Queue()
	completed = 0

	def _worker_wrapper(job_i, queue):
		result = process_one_file(job_i)
		queue.put(result)

	n_parallel = global_threats
	active_procs = []


	start_processing = datetime.datetime.now()
	# Start multiple threads
	for i, job_i in enumerate(all_jobs):
		proc = mp.Process(target=_worker_wrapper, args=(job_i, result_queue))
		proc.start()
		active_procs.append(proc)

		# Until limit hit
		if len(active_procs) >= n_parallel:
			for p in active_procs:
				p.join()
			while not result_queue.empty():
				result = result_queue.get()
				if result is not None:
					nameArray.append(result[0])
					resGlobArray.append(result[1])
					ratioSignalArray.append(result[2])
			completed += len(active_procs)
			print(f"Completed {completed}/{it_loops}")
			active_procs = []
			end_batch = datetime.datetime.now()
			current_time = end_batch-start_processing
			print("Estimated remaining time: " + str((current_time/completed)*(nFiles-completed)) + "\n\n")

	# Collect any remaining
	for p in active_procs:
		p.join()
	while not result_queue.empty():
		result = result_queue.get()
		if result is not None:
			nameArray.append(result[0])
			resGlobArray.append(result[1])
			ratioSignalArray.append(result[2])
	completed += len(active_procs)
	print(f"Completed {completed}/{it_loops}")
  
	# Write summary.tsv
	if len(nameArray) > 0 and config != "Refined-Maps":
		with open(os.path.join(outputDir, "summary.tsv"), 'w', encoding='utf-8') as file:
			file.write('id\tmedian_resolution\tsignal_ratio\n')
			nameArray = np.array(nameArray)
			resGlobArray = np.array(resGlobArray)
			ratioSignalArray = np.array(ratioSignalArray)
			sortedIndices = np.argsort(resGlobArray)
			nameArray = nameArray[sortedIndices]
			resGlobArray = resGlobArray[sortedIndices]
			ratioSignalArray = ratioSignalArray[sortedIndices]
			for i in range(len(nameArray)):
				file.write(f'{nameArray[i]}\t{resGlobArray[i]}\t{ratioSignalArray[i]}\n')

	print(f"\nBatch complete. Processed {len(nameArray)} files successfully.")
	print("IN TOTAL: " + str(datetime.datetime.now()-start_total) + "\n\n")

if __name__ == '__main__':    
	main()

