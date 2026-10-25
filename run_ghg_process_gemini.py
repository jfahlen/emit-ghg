#!/usr/bin/env python3

import subprocess
import argparse
import glob
import os
import json
import shlex
import pdb

# --- Global Configurations ---
CONDA_ENV = "conda activate emit-ghg; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export PYTHONPATH=~/src/py_tools:~/src/py_tools/emit/:~/src/plume_emission_rate:~/src/hapi/:~/src/SpectralUtil/spectral_util/;"
EMIT_OPS_BASE = "/store/emit/ops/data/acquisitions"
KURUCZ_FILE = "/store/airborne/software/asds_data/main/l1b_rdn/kurucz_0.1nm.dat"
NOISE_FILE_AV3 = "/home/jfahlen/src/emit-ghg/instrument_noise_parameters/AV320250715t180536_001_L1B_RDN_76a54582_NOISE.txt"

def submit_job(fid, python_commands, output_dir, execute, cmd_type='srun', cores="1", mem="30G"):
    """Wraps and submits the python command(s) via srun or sbatch."""
    
    # Join multiple python commands with a semicolon if necessary
    if isinstance(python_commands[0], list):
        cmd_str = " ; ".join([shlex.join(c) for c in python_commands])
    else:
        cmd_str = shlex.join(python_commands)

    if cmd_type == 'sbatch':
        full_command = [
            "sbatch", "-N", "1", "-c", cores, f"--mem={mem}",
            "-o", f"{output_dir}/log.%j.out",
            "--wrap", f"{CONDA_ENV} {cmd_str}"
        ]
    else:  # srun
        # Use bash -c to allow sequential execution and environment setup if needed
        full_command = [
            "srun", "-p", "debug", "-N", "1", "-c", cores, f"--mem={mem}", "--pty",
            "bash", "-c", f"{CONDA_ENV} {cmd_str}"
        ]

    print("Command to be executed:")
    print(shlex.join(full_command))

    if execute:
        try:
            subprocess.run(full_command, check=True)
            print(f"Command executed successfully for case prefix {fid}.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for case prefix {fid}: {e}")
    else:
        print("Use --execute to run the command.")

def run_ghg_process_EMIT_cluster(fid, output_dir, wavelength_range, execute=False, cmd='srun'):
    case_prefix = fid[4:]
    date_str = case_prefix[:8]
    time_str = case_prefix[9:15]

    search_pattern = f"{EMIT_OPS_BASE}/{date_str}/emit{date_str}t{time_str}/l1b/emit{date_str}t{time_str}*l1b_rdn*.img"
    files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {search_pattern}")

    filename = os.path.basename(files[0])
    parts = filename.split("_")
    orbit, scene = parts[1], parts[2]

    base_path = f"{EMIT_OPS_BASE}/{date_str}/emit{date_str}t{time_str}"
    
    #ghg_process_command = [
    #    "python", "emit-ghg/deploy/ghg_process.py",
    #    f"{base_path}/l1b/emit{date_str}t{time_str}_{orbit}_{scene}_l1b_rdn_b0106_v01.img",
    #    f"{base_path}/l1b/emit{date_str}t{time_str}_{orbit}_{scene}_l1b_obs_b0106_v01.img",
    #    f"{base_path}/l1b/emit{date_str}t{time_str}_{orbit}_{scene}_l1b_loc_b0106_v01.img",
    #    f"{base_path}/l1b/emit{date_str}t{time_str}_{orbit}_{scene}_l1b_glt_b0106_v01.img",
    #    f"{base_path}/l1b/emit{date_str}t{time_str}_{orbit}_{scene}_l1b_bandmask_b0106_v01.img",
    #    f"{base_path}/l2a/emit{date_str}t{time_str}_{orbit}_{scene}_l2a_mask_b0106_v01.img",
    #    os.path.join(output_dir, fid),
    #    "--wavelength_range"
    #] + [str(int(w)) for w in wavelength_range] + [
    #    "--state_subs", f"{base_path}/l2a/emit{date_str}t{time_str}_{orbit}_{scene}_l2a_statesubs_b0106_v01.img",
    #    "--overwrite"
    #]

    ghg_process_command = [
        "python", "deploy/ghg_process.py",
        #"python", "emit-ghg/deploy/ghg_process.py",
        f"{base_path}/l1b/emit{date_str}t{time_str}_l1b_rdn_v02.img",
        f"{base_path}/l1b/emit{date_str}t{time_str}_l1b_obs_v02.img",
        f"{base_path}/l1b/emit{date_str}t{time_str}_l1b_loc_v02.img",
        f"{base_path}/l1b/emit{date_str}t{time_str}_l1b_glt_v02.img",
        f"{base_path}/l1b/emit{date_str}t{time_str}_l1b_bandmask_v02.img",
        f"{base_path}/mask/emit{date_str}t{time_str}_l2a_maskTf_v03.img",
        os.path.join(output_dir, fid),
        "--noise_file", "/home/jfahlen/src/emit-ghg/emit_ghg/data/instrument_noise_parameters/emit_noise.txt"
    ]

    if len(wavelength_range) > 0:
        ghg_process_command.extend("--wavelength_range" + [str(int(w)) for w in wavelength_range])
        
        
    ghg_process_command.extend([
        "--state_subs", f"{base_path}/l2a/emit{date_str}t{time_str}_l2a_statesubs_v02.img",
        "--overwrite"
    ])

    submit_job(fid, ghg_process_command, output_dir, execute, cmd_type=cmd, cores="1", mem="30G")

def run_ghg_process_EMIT_DAAC(fid, input_path, output_dir, execute=False, cmd='srun'):
    json_file_list_filename = os.path.join(input_path, 'data_files.json')
    if not os.path.exists(json_file_list_filename):
        raise FileNotFoundError(f'Could not find {json_file_list_filename}')
    
    with open(json_file_list_filename) as f:
        json_file_list = json.load(f)

    ghg_process_command = [
        "python", "ghg_process.py",
        json_file_list['RAD'], json_file_list['OBS'],
        json_file_list['RAD'], json_file_list['OBS'],
        json_file_list['L2A_MASK'], json_file_list['L2A_MASK'],
        output_dir,
        "--state_subs", json_file_list['L2A_MASK'],
        "--overwrite"
    ]

    submit_job(fid, ghg_process_command, output_dir, execute, cmd_type=cmd, cores="1", mem="30G")

def run_ghg_process_AV3_DAAC(fid, input_path, output_dir, wavelength_range, execute=False, cmd='srun'):
    json_file_list_filename = os.path.join(input_path, 'data_files.json')
    if not os.path.exists(json_file_list_filename):
        raise FileNotFoundError(f'Could not find {json_file_list_filename}')
        
    with open(json_file_list_filename) as f:
        json_file_list = json.load(f)

    l1b_rdn_nc_search_pattern = f'{input_path}/*_RDN.nc'
    l1b_rdn_files = glob.glob(l1b_rdn_nc_search_pattern)
    if len(l1b_rdn_files) != 1:
        raise ValueError(f'Found {len(l1b_rdn_files)} folders at {l1b_rdn_nc_search_pattern}')
    
    l1b_rdn_file = l1b_rdn_files[0]
    l1b_mask_filename = os.path.join(output_dir, 'L1B_MASK')

    l1b_mask_command = [
        "python", 'make_emit_masks.py',
        l1b_rdn_file, l1b_rdn_file,
        KURUCZ_FILE, l1b_mask_filename
    ]

    ghg_process_command = [
        "python", "ghg_process.py",
        l1b_rdn_file, json_file_list['OBS'],
        l1b_rdn_file, json_file_list['OBS'],
        json_file_list['BANDMASK'], l1b_mask_filename,
        output_dir,
        "--wavelength_range"
    ] + [str(int(w)) for w in wavelength_range] + [
        "--overwrite",
        '--noise_file', NOISE_FILE_AV3
    ]

    submit_job(fid, [l1b_mask_command, ghg_process_command], output_dir, execute, cmd_type=cmd, cores="64", mem="300G")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct and optionally execute ghg_process.py with case prefix and output directory.")
    
    parser.add_argument("--cmd", choices=["srun", "sbatch"], default="srun", help="Execution command to use (default: srun)")

    subparsers = parser.add_subparsers(dest="mode", title="mode", description="run mode", required=True, help="choose which processing mode to run")

    parser_emit = subparsers.add_parser("EMIT_cluster", help="Run EMIT processing")
    parser_emit.add_argument("fid", help="The case prefix (e.g., emit20250623t173722)")
    parser_emit.add_argument("output_dir", help="The output directory")
    #parser_emit.add_argument("--wavelengths", type=float, nargs="+", help="One or more min max pairs")
    parser_emit.add_argument("--wavelengths", type=float, nargs="+", default=[], help="One or more min max pairs",)
    parser_emit.add_argument("--execute", action="store_true", help="Execute the command")

    parser_av3 = subparsers.add_parser("AV3_DAAC", help="Run AV3 processing")
    parser_av3.add_argument("fid", help="The case prefix (e.g., AV320240914t203535)")
    parser_av3.add_argument("input_path", help="Path to the input data granule")
    parser_av3.add_argument("output_dir", help="The output directory")
    parser_av3.add_argument("--wavelengths", type=float, nargs="+", help="One or more min max pairs")
    parser_av3.add_argument("--execute", action="store_true", help="Execute the command")

    parser_emit_daac = subparsers.add_parser("EMIT_DAAC", help="Run EMIT DAAC processing")
    parser_emit_daac.add_argument("fid", help="The case prefix (e.g., emit20250623t173722)")
    parser_emit_daac.add_argument("input_path", help="Path to the input data granule")
    parser_emit_daac.add_argument("output_dir", help="The output directory")
    parser_emit_daac.add_argument("--execute", action="store_true", help="Execute the command")

    args = parser.parse_args()

    if args.mode == 'EMIT_cluster':
        run_ghg_process_EMIT_cluster(args.fid, args.output_dir, args.wavelengths, args.execute, cmd=args.cmd)
    elif args.mode == 'EMIT_DAAC':
        run_ghg_process_EMIT_DAAC(args.fid, args.input_path, args.output_dir, args.execute, cmd=args.cmd)
    elif args.mode == 'AV3_DAAC':
        run_ghg_process_AV3_DAAC(args.fid, args.input_path, args.output_dir, args.wavelengths, args.execute, cmd=args.cmd)