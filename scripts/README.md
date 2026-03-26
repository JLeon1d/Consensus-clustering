# Remote Server Scripts

Scripts for running benchmarks on remote server and setting up distributed Ray cluster.

## Basic Remote Execution

1. Create `.env` file in project root with credentials
2. Run setup: `python3 setup_remote_ray.py`
3. Upload code: `python3 upload_code.py`
4. Run benchmarks: `python3 run_remote_benchmarks.py`

## Distributed Ray Cluster

Use both local and remote machines for parallel processing:

```bash
# 1. Start Ray head node locally
ray start --head --port=6379

# 2. Setup remote worker
python3 setup_ray_cluster.py setup

# 3. Check cluster status
python3 setup_ray_cluster.py status

# 4. Run benchmarks (Ray will use both machines)
cd ../benchmarks && python3 run_all_benchmarks.py

# 5. Stop cluster when done
python3 setup_ray_cluster.py stop
ray stop  # Stop local head node
```

## Scripts

- `remote_utils.py` - SSH connection utilities
- `setup_remote_ray.py` - Install Ray on remote server
- `upload_code.py` - Upload project code
- `run_remote_benchmarks.py` - Execute benchmarks remotely
- `setup_ray_cluster.py` - Setup distributed Ray cluster (local + remote)

## Network Requirements for Distributed Ray

- Port 6379 must be accessible from remote server to local machine
- Firewall must allow incoming connections on port 6379
- Both machines must be able to reach each other
