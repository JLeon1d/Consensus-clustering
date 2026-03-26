"""Upload project code to remote server."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from remote_utils import RemoteServer


def upload_project_code():
    """Upload necessary project files to remote server."""
    
    with RemoteServer() as server:
        print("\n" + "="*70)
        print("Uploading project code to remote server")
        print("="*70 + "\n")
        
        project_root = Path(__file__).parent.parent
        remote_dir = "~/consensus_clustering_benchmarks/code"
        
        # Create remote directory structure
        print("Creating remote directories...")
        server.execute_command(f"mkdir -p {remote_dir}/src")
        server.execute_command(f"mkdir -p {remote_dir}/benchmarks")
        
        # Upload source code
        print("\nUploading source code...")
        src_dir = project_root / "src"
        if src_dir.exists():
            server.upload_directory(str(src_dir), f"{remote_dir}/src")
        
        # Upload benchmark scripts (we'll create new ones)
        print("\nUploading benchmark scripts...")
        benchmarks_dir = project_root / "benchmarks"
        if benchmarks_dir.exists():
            for file in benchmarks_dir.glob("*.py"):
                remote_file = f"{remote_dir}/benchmarks/{file.name}"
                server.upload_file(str(file), remote_file)
        
        # Upload requirements
        print("\nUploading requirements...")
        req_file = project_root / "requirements.txt"
        if req_file.exists():
            server.upload_file(str(req_file), f"{remote_dir}/requirements.txt")
        
        # Install project dependencies
        print("\nInstalling project dependencies...")
        install_cmd = " && ".join([
            "cd ~/consensus_clustering_benchmarks",
            "source venv/bin/activate",
            "cd code",
            "pip install -e ."
        ])
        server.execute_command(install_cmd)
        
        print("\n" + "="*70)
        print("✓ Code upload completed!")
        print("="*70)


if __name__ == "__main__":
    upload_project_code()
