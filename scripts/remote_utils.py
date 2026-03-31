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
                timeout=10,
                allow_agent=False,
                look_for_keys=False
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
        
        # Expand ~ in remote path
        if remote_path.startswith('~/'):
            stdout, stderr, code = self.execute_command('echo $HOME', verbose=False)
            home_dir = stdout.strip()
            remote_path = remote_path.replace('~', home_dir, 1)
        
        self.sftp.put(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: str):
        """Download file from remote server."""
        if not self.sftp:
            raise RuntimeError("SFTP not initialized")
        
        # Expand ~ in remote path
        if remote_path.startswith('~/'):
            stdout, stderr, code = self.execute_command('echo $HOME', verbose=False)
            home_dir = stdout.strip()
            remote_path = remote_path.replace('~', home_dir, 1)

        self.sftp.get(remote_path, local_path)

    def upload_directory(self, local_dir: str, remote_dir: str, skip_dirs: set = None):
        """Upload directory to remote server, skipping specified directories."""
        if not self.sftp:
            raise RuntimeError("SFTP not initialized")

        if skip_dirs is None:
            skip_dirs = {'__pycache__', '.pytest_cache', '.git'}

        # Expand ~ in remote path
        if remote_dir.startswith('~/'):
            stdout, stderr, code = self.execute_command('echo $HOME', verbose=False)
            home_dir = stdout.strip()
            remote_dir = remote_dir.replace('~', home_dir, 1)
        
        local_path = Path(local_dir)

        # Create remote directory recursively
        self._mkdir_p(remote_dir)

        for item in local_path.rglob('*'):
            # Skip if any parent directory is in skip_dirs
            if any(part in skip_dirs for part in item.parts):
                continue

            if item.is_file():
                relative_path = item.relative_to(local_path)
                remote_file = f"{remote_dir}/{relative_path}".replace('\\', '/')

                # Create parent directories recursively
                remote_parent = '/'.join(remote_file.split('/')[:-1])
                self._mkdir_p(remote_parent)
                
                print(f"Uploading {item.name}")
                self.sftp.put(str(item), remote_file)

    def _mkdir_p(self, remote_path: str):
        """Create directory recursively on remote server."""
        if not self.sftp:
            return
        
        dirs = []
        path = remote_path
        while path and path != '/':
            dirs.append(path)
            path = '/'.join(path.split('/')[:-1])
        
        for dir_path in reversed(dirs):
            try:
                self.sftp.stat(dir_path)
            except IOError:
                try:
                    self.sftp.mkdir(dir_path)
                except IOError:
                    pass

    def close(self):
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        print("Connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
