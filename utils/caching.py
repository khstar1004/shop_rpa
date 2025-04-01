import os
import pickle
import time
from typing import Any, Optional, Callable, List, Tuple
import hashlib
import logging
import zlib
import random
import shutil

logger = logging.getLogger(__name__)

class FileCache:
    """Simple file-based cache with expiration and size management."""
    
    def __init__(self, cache_dir: str, duration_seconds: int = 86400, max_size_mb: int = 1024,
                 enable_compression: bool = False, compression_level: int = 6):
        """
        Args:
            cache_dir: Directory to store cache files.
            duration_seconds: Cache duration in seconds (default: 1 day).
            max_size_mb: Maximum size of cache directory in MB (default: 1GB).
            enable_compression: Whether to compress cache entries.
            compression_level: Compression level (1-9, default: 6).
        """
        self.cache_dir = cache_dir
        self.duration = duration_seconds
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check current cache size on initialization
        cache_size = self._get_cache_size()
        logger.info(
            f"Cache initialized at '{self.cache_dir}' with duration {duration_seconds} seconds. "
            f"Current size: {cache_size / 1024 / 1024:.2f}MB, Max size: {max_size_mb}MB. "
            f"Compression {'enabled' if enable_compression else 'disabled'}."
        )
        
        # Clean up if cache is too large
        if cache_size > self.max_size_bytes:
            self._cleanup_cache()

    def _get_cache_filepath(self, key: str) -> str:
        """Generate a safe filename for the cache key."""
        # Use SHA256 hash for a consistent and safe filename
        hashed_key = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.pkl")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from the cache if it exists and is not expired."""
        filepath = self._get_cache_filepath(key)
        if not os.path.exists(filepath):
            logger.debug(f"Cache miss (file not found) for key: {key[:50]}...")
            return None

        try:
            with open(filepath, 'rb') as f:
                data = f.read()
                
            # Uncompress if compression is enabled
            if self.enable_compression:
                try:
                    data = zlib.decompress(data)
                except zlib.error as e:
                    logger.warning(f"Failed to decompress cache entry: {e}. Treating as uncompressed.")
                    # Assume it's not compressed and continue
                
            data, timestamp = pickle.loads(data)
            
            if time.time() - timestamp > self.duration:
                logger.debug(f"Cache miss (expired) for key: {key[:50]}...")
                os.remove(filepath) # Remove expired file
                return None
            
            # Update file access time to mark it as recently used
            os.utime(filepath, None)
            
            logger.debug(f"Cache hit for key: {key[:50]}...")
            return data
        except (OSError, pickle.PickleError, EOFError) as e:
            logger.warning(f"Error reading cache file {filepath} for key '{key[:50]}...': {e}")
            # Attempt to remove corrupted cache file
            try:
                os.remove(filepath)
            except OSError:
                pass
            return None

    def set(self, key: str, value: Any) -> None:
        """Store an item in the cache."""
        # First check if we need to clean up the cache
        if self._should_cleanup_cache():
            self._cleanup_cache()
            
        filepath = self._get_cache_filepath(key)
        try:
            # Serialize the data with timestamp
            data = pickle.dumps((value, time.time()))
            
            # Compress if enabled
            if self.enable_compression:
                data = zlib.compress(data, level=self.compression_level)
            
            # Write atomically to prevent partial writes
            temp_filepath = filepath + ".tmp"
            with open(temp_filepath, 'wb') as f:
                f.write(data)
            os.replace(temp_filepath, filepath) # Atomic rename/replace
            logger.debug(f"Cache set for key: {key[:50]}...")
        except (OSError, pickle.PickleError) as e:
            logger.error(f"Error writing cache file {filepath} for key '{key[:50]}...': {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass

    def clear(self) -> None:
        """Clear the entire cache."""
        logger.info(f"Clearing cache directory: {self.cache_dir}")
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            try:
                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    # Optionally clear subdirectories if needed
                    pass
            except Exception as e:
                logger.error(f'Failed to delete {filepath}. Reason: {e}')
        logger.info("Cache cleared.")

    def _get_cache_size(self) -> int:
        """Get the current size of the cache in bytes."""
        total_size = 0
        for dirpath, _, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size

    def _should_cleanup_cache(self) -> bool:
        """Check if cache cleanup is needed based on size and a random probability."""
        # Check the size with a 10% random chance to avoid checking too frequently
        if random.random() < 0.1:
            cache_size = self._get_cache_size()
            return cache_size > self.max_size_bytes * 0.9  # Clean up at 90% of max size
        return False
        
    def _cleanup_cache(self) -> None:
        """Clean up cache when it exceeds the maximum size limit."""
        logger.info(f"Starting cache cleanup. Current size exceeds limit.")
        
        # List all cache files with their access time and size
        cache_files = []
        for dirpath, _, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath) and filepath.endswith('.pkl'):
                    try:
                        stats = os.stat(filepath)
                        cache_files.append((
                            filepath,
                            stats.st_atime,  # Access time
                            stats.st_size    # File size in bytes
                        ))
                    except OSError:
                        # Skip files that can't be stat'd
                        pass
        
        if not cache_files:
            logger.info("No cache files found to clean up.")
            return
            
        # Sort files by access time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Calculate current cache size
        current_size = sum(item[2] for item in cache_files)
        target_size = self.max_size_bytes * 0.7  # Reduce to 70% of max size
        
        logger.info(f"Current cache size: {current_size / 1024 / 1024:.2f}MB, "
                   f"Target size: {target_size / 1024 / 1024:.2f}MB")
        
        # Delete oldest files until we're under the target size
        deleted_count = 0
        deleted_size = 0
        
        for filepath, _, size in cache_files:
            if current_size <= target_size:
                break
                
            try:
                os.remove(filepath)
                current_size -= size
                deleted_size += size
                deleted_count += 1
            except OSError as e:
                logger.warning(f"Failed to delete cache file {filepath}: {e}")
        
        logger.info(f"Cache cleanup completed. Deleted {deleted_count} files "
                   f"({deleted_size / 1024 / 1024:.2f}MB). "
                   f"New size: {current_size / 1024 / 1024:.2f}MB")

# Decorator for function caching
def cache_result(cache: Optional[FileCache], key_prefix: str = "func") -> Callable:
    """Decorator to cache the result of a function."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # If no cache is provided, just execute the function without caching
            if cache is None:
                return func(*args, **kwargs)
                
            # Generate cache key from function name, args, and kwargs
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(map(str, args))
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = "|".join(key_parts)
            
            # Check cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        return wrapper
    return decorator 