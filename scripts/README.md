# Remote Server Scripts

Scripts for running benchmarks on remote server.

## Setup

1. Create `.env` file in project root with credentials
2. Run setup: `python3 setup_remote_ray.py`
3. Upload code: `python3 upload_code.py`
4. Run benchmarks: `python3 run_remote_benchmarks.py`

## Scripts

- `remote_utils.py` - SSH connection utilities
- `setup_remote_ray.py` - Install Ray on remote server
- `upload_code.py` - Upload project code
- `run_remote_benchmarks.py` - Execute benchmarks and download results
