"""Setup Ray cluster with local machine as head and remote as worker."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from remote_utils import RemoteServer


def setup_ray_cluster():
    """
    Setup Ray cluster with local machine as head node and remote server as worker.
    
    This allows Ray to distribute work across both machines.
    """
    
    print("="*70)
    print("Setting up Ray Cluster (Local Head + Remote Worker)")
    print("="*70)
    
    # Get local IP address
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\nLocal machine IP: {local_ip}")
    
    with RemoteServer() as server:
        print("\n1. Installing Ray on remote server...")
        
        # Ensure Ray is installed
        install_cmd = " && ".join([
            "cd ~/consensus_clustering_benchmarks",
            "source venv/bin/activate",
            "pip install 'ray[default]>=2.9.0'"
        ])
        server.execute_command(install_cmd)
        
        print("\n2. Starting Ray worker on remote server...")
        print(f"   Connecting to head node at: {local_ip}:6379")
        
        # Start Ray worker that connects to local head node
        # Note: You'll need to start the head node locally first
        worker_cmd = " && ".join([
            "cd ~/consensus_clustering_benchmarks",
            "source venv/bin/activate",
            f"ray start --address='{local_ip}:6379' --num-cpus=4"
        ])
        
        stdout, stderr, code = server.execute_command(worker_cmd)
        
        if code != 0:
            print(f"✗ Failed to start Ray worker")
            print(f"Error: {stderr}")
            print("\nMake sure:")
            print("1. Ray head node is running locally: ray start --head --port=6379")
            print(f"2. Port 6379 is accessible from remote server to {local_ip}")
            print("3. Firewall allows incoming connections on port 6379")
            return False
        
        print("✓ Ray worker started on remote server")
        print("\nCluster setup complete!")
        print("\nTo use the cluster:")
        print("1. Start Ray head locally: ray start --head --port=6379")
        print("2. Run benchmarks with Ray enabled")
        print("3. Ray will automatically distribute work to remote worker")
        
        return True


def stop_ray_cluster():
    """Stop Ray worker on remote server."""
    
    print("\nStopping Ray worker on remote server...")
    
    with RemoteServer() as server:
        stop_cmd = " && ".join([
            "cd ~/consensus_clustering_benchmarks",
            "source venv/bin/activate",
            "ray stop"
        ])
        
        server.execute_command(stop_cmd)
        print("✓ Ray worker stopped")


def show_cluster_status():
    """Show Ray cluster status."""
    
    print("\n" + "="*70)
    print("Ray Cluster Status")
    print("="*70)
    
    # Local status
    print("\nLocal (Head Node):")
    os.system("ray status 2>/dev/null || echo 'Ray not running locally'")
    
    # Remote status
    print("\nRemote (Worker Node):")
    with RemoteServer() as server:
        status_cmd = " && ".join([
            "cd ~/consensus_clustering_benchmarks",
            "source venv/bin/activate",
            "ray status"
        ])
        
        stdout, stderr, code = server.execute_command(status_cmd, verbose=False)
        if code == 0:
            print(stdout)
        else:
            print("Ray not running on remote server")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Ray cluster")
    parser.add_argument('action', choices=['setup', 'stop', 'status'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'setup':
        setup_ray_cluster()
    elif args.action == 'stop':
        stop_ray_cluster()
    elif args.action == 'status':
        show_cluster_status()
