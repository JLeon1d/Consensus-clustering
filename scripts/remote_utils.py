"""Utilities for remote server connection and management."""

import os
import sys
from pathlib import Path
from typing import Optional

import paramiko
from dotenv import load_dotenv


class RemoteServer:
    """Manage SSH connection to remote server."""

    def __init__(self):
        """Initialize remote server connection from .env file."""
        # Load environment variables
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)

        self.host = os.getenv("REMOTE_HOST")
        self.port = int(os.getenv("REMOTE_PORT", "22"))
        self.user = os.getenv("REMOTE_USER")
        self.password = os.getenv("REMOTE_PASSWORD")

        if not all([self.host, self.user, self.password]):
            raise ValueError("Missing remote server credentials in .env file")

        self.client: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None

    def connect(self):
        """Establish SSH connection to remote server."""
        print(f"Connecting to {self.user}@{self.host}:{self.port}...")
        
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            self.client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                timeout=10
            )
            self.sftp = self.client.open_sftp()
            print("✓ Connected successfully!")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def execute_command(self, command: str, verbose: bool = True) -> tuple[str, str, int]:
        """
        Execute command on remote server.
        
        Returns:
            tuple: (stdout, stderr, exit_code)
        """
        if not self.client:
            raise RuntimeError("Not connected to remote server")

        if verbose:
            print(f"Executing: {command}")
        
        stdin, stdout, stderr = self.client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        
        stdout_text = stdout.read().decode('utf-8')
        stderr_text = stderr.read().decode('utf-8')
        
        if verbose and stdout_text:
            print(stdout_text)
        if verbose and stderr_text:
            print(f"STDERR: {stderr_text}", file=sys.stderr)
        
        return stdout_text, stderr_text, exit_code

    def upload_file(self, local_path: str, remote_path: str):
        """Upload file to remote server."""
        if not self.sftp:
            raise RuntimeError("SFTP not initialized")
        
        print(f"Uploading {local_path} -> {remote_path}")
        self.sftp.put(local_path, remote_path)
        print("✓ Upload complete")

    def download_file(self, remote_path: str, local_path: str):
        """Download file from remote server."""
        if not self.sftp:
            raise RuntimeError("SFTP not initialized")
        
        print(f"Downloading {remote_path} -> {local_path}")
        self.sftp.get(remote_path, local_path)
        print("✓ Download complete")

    def upload_directory(self, local_dir: str, remote_dir: str):
        """Upload directory to remote server."""
        if not self.sftp:
            raise RuntimeError("SFTP not initialized")
        
        local_path = Path(local_dir)
        
        # Create remote directory
        try:
            self.sftp.mkdir(remote_dir)
        except IOError:
            pass  # Directory might already exist
        
        for item in local_path.rglob('*'):
            if item.is_file():
                relative_path = item.relative_to(local_path)
                remote_file = f"{remote_dir}/{relative_path}".replace('\\', '/')
                
                # Create parent directories
                remote_parent = '/'.join(remote_file.split('/')[:-1])
                try:
                    self.sftp.mkdir(remote_parent)
                except IOError:
                    pass
                
                print(f"Uploading {item} -> {remote_file}")
                self.sftp.put(str(item), remote_file)

    def close(self):
        """Close SSH connection."""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        print("Connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def test_connection():
    """Test connection to remote server."""
    with RemoteServer() as server:
        stdout, stderr, code = server.execute_command("uname -a")
        print(f"Exit code: {code}")
        
        stdout, stderr, code = server.execute_command("python3 --version")
        print(f"Exit code: {code}")


if __name__ == "__main__":
    test_connection()
