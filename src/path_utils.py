import os
from pathlib import Path

def resolve_data_root() -> Path:
    """
    Determine the writable data root inside the container.
    Checks HOST_DATA, CONTAINER_DATA_ROOT env vars and common mount points.
    """
    raw = (os.environ.get('HOST_DATA') or os.environ.get('CONTAINER_DATA_ROOT') or '').strip()
    candidates: list[str] = []
    if raw:
        # Try to handle Windows paths passed as env vars
        if ':' in raw and '\\' in raw:
            drive = raw[0].lower()
            rest = raw[2:].replace('\\', '/')
            candidates.append(f"/host_mnt/{drive}/{rest}")
            candidates.append(f"/mnt/{drive}/{rest}")
        candidates.append(raw)
    
    candidates.append('/data')
    
    for cand in candidates:
        try:
            path = Path(cand)
            if path.is_dir():
                return path.resolve()
            # Try to create if it looks like a valid path we want to use
            # But usually we look for existing mounts. 
            # If we are in a container, /data might exist but be empty.
            if path.parent.exists():
                 path.mkdir(parents=True, exist_ok=True)
                 return path.resolve()
        except Exception:
            continue
            
    # Fallback
    return Path('/data').resolve()

def convert_windows_to_wsl(p: str) -> str:
    """
    Convert a Windows path (e.g. C:\Data) to a WSL/Linux path (e.g. /mnt/c/Data).
    """
    if not isinstance(p, str):
        return p
    if len(p) >= 3 and p[1:3] == ':\\':
        drive = p[0].lower()
        rest = p[3:].replace('\\', '/')
        return f"/mnt/{drive}/{rest}"
    return p

def map_path_for_container(p: str) -> str:
    """
    Map incoming paths for the current runtime (WSL/Linux):
    - Windows-style paths (e.g. C:\...) -> /mnt/<drive>/...
    - Leave existing POSIX paths unchanged.
    """
    if not isinstance(p, str):
        return p
    # Windows path -> WSL path
    if '\\' in p or (len(p) >= 3 and p[1:3] == ':\\'):
        return convert_windows_to_wsl(p)
    # Already POSIX -> leave as is
    return p
