"""Setup Ray on remote server."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from remote_utils import RemoteServer


def setup_ray_on_remote():
    """Install and configure Ray on remote server."""
    
    with RemoteServer() as server:
        print("\n" + "="*70)
        print("Setting up Ray on remote server")
        print("="*70 + "\n")
        
        # Check Python version
        print("1. Checking Python version...")
        stdout, stderr, code = server.execute_command("python3 --version")
        if code != 0:
            print("✗ Python3 not found!")
            return False
        
        # Check if pip is installed
        print("\n2. Checking pip...")
        stdout, stderr, code = server.execute_command("python3 -m pip --version")
        if code != 0:
            print("Installing pip...")
            server.execute_command("curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py")
            server.execute_command("python3 get-pip.py --user")
            server.execute_command("rm get-pip.py")
        
        # Create project directory
        print("\n3. Creating project directory...")
        server.execute_command("mkdir -p ~/consensus_clustering_benchmarks")
        
        # Install virtualenv if not present
        print("\n4. Setting up virtual environment...")
        server.execute_command("python3 -m pip install --user virtualenv")
        
        # Create virtual environment
        server.execute_command("cd ~/consensus_clustering_benchmarks && python3 -m venv venv")
        
        # Install dependencies
        print("\n5. Installing dependencies...")
        commands = [
            "cd ~/consensus_clustering_benchmarks",
            "source venv/bin/activate",
            "pip install --upgrade pip",
            "pip install numpy scipy scikit-learn pandas",
            "pip install 'ray[default]>=2.9.0'",
            "pip install python-dotenv paramiko",
        ]
        
        cmd = " && ".join(commands)
        stdout, stderr, code = server.execute_command(cmd)
        
        if code != 0:
            print("✗ Failed to install dependencies")
            return False
        
        # Verify Ray installation
        print("\n6. Verifying Ray installation...")
        verify_cmd = " && ".join([
            "cd ~/consensus_clustering_benchmarks",
            "source venv/bin/activate",
            "python3 -c 'import ray; print(f\"Ray version: {ray.__version__}\")'"
        ])
        stdout, stderr, code = server.execute_command(verify_cmd)
        
        if code != 0:
            print("✗ Ray verification failed")
            return False
        
        print("\n" + "="*70)
        print("✓ Ray setup completed successfully!")
        print("="*70)
        return True


if __name__ == "__main__":
    success = setup_ray_on_remote()
    sys.exit(0 if success else 1)
