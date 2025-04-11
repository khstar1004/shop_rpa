"""Provides caching utilities, including a file-based cache and a function result caching decorator."""
import hashlib
import logging
import os
import pickle
import random
import time
import zlib
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class FileCache:
    """Simple file-based cache with expiration and size management."""

    def __init__(
        self,
        cache_dir: str,
        duration_seconds: int = 86400,
        max_size_mb: int = 1024,
        enable_compression: bool = False,
        compression_level: int = 6,
    ):
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
            "Cache initialized at '%s' with duration %d seconds. "
            "Current size: %.2fMB, Max size: %dMB. Compression %s.",
            self.cache_dir, duration_seconds,
            cache_size / 1024 / 1024, max_size_mb,
            'enabled' if enable_compression else 'disabled'
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
            logger.debug("Cache miss (file not found) for key: %s...", key[:50])
            return None

        try:
            with open(filepath, "rb") as f:
                data = f.read()

            # Uncompress if compression is enabled
            if self.enable_compression:
                try:
                    data = zlib.decompress(data)
                except zlib.error as e:
                    logger.warning(
                        "Failed to decompress cache entry: %s. Treating as uncompressed.", e
                    )
                    # Assume it's not compressed and continue

            data, expiration_time = pickle.loads(data)

            if time.time() > expiration_time:
                logger.debug("Cache miss (expired) for key: %s...", key[:50])
                os.remove(filepath)  # Remove expired file
                return None

            # Update file access time to mark it as recently used
            os.utime(filepath, None)

            logger.debug("Cache hit for key: %s...", key[:50])
            return data
        except (OSError, pickle.PickleError, EOFError) as e:
            logger.warning(
                "Error reading cache file %s for key '%s...': %s",
                filepath, key[:50], e
            )
            # Attempt to remove corrupted cache file
            try:
                os.remove(filepath)
            except OSError:
                pass
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store an item in the cache.

        Args:
            key: Unique identifier for the cache entry
            value: The data to cache
            ttl: Optional time-to-live in seconds. If provided, this overrides the default duration.
        """
        # First check if we need to clean up the cache
        if self._should_cleanup_cache():
            self._cleanup_cache()

        filepath = self._get_cache_filepath(key)
        try:
            # Use provided ttl or default duration
            expiration_time = time.time() + (ttl if ttl is not None else self.duration)

            # Serialize the data with timestamp
            data = pickle.dumps((value, expiration_time))

            # Compress if enabled
            if self.enable_compression:
                data = zlib.compress(data, level=self.compression_level)

            # Write atomically to prevent partial writes
            temp_filepath = filepath + ".tmp"
            with open(temp_filepath, "wb") as f:
                f.write(data)
            os.replace(temp_filepath, filepath)  # Atomic rename/replace
            logger.debug("Cache set for key: %s...", key[:50])
        except (OSError, pickle.PickleError) as e:
            logger.error(
                "Error writing cache file %s for key '%s...': %s",
                filepath, key[:50], e
            )
            # Clean up temp file if it exists
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass

    def clear(self) -> None:
        """Clear the entire cache."""
        logger.info("Clearing cache directory: %s", self.cache_dir)
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            try:
                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    # Optionally clear subdirectories if needed
                    pass
            except OSError as e:
                logger.error("Failed to delete %s. Reason: %s", filepath, e)
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
        logger.info("Starting cache cleanup. Current size exceeds limit.")

        # List all cache files with their access time and size
        cache_files = []
        for dirpath, _, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath) and filepath.endswith(".pkl"):
                    try:
                        stats = os.stat(filepath)
                        cache_files.append(
                            (
                                filepath,
                                stats.st_atime,  # Access time
                                stats.st_size,  # File size in bytes
                            )
                        )
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

        logger.info(
            "Current cache size: %.2fMB, Target size: %.2fMB",
            current_size / 1024 / 1024, target_size / 1024 / 1024
        )

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
                logger.warning("Failed to delete cache file %s: %s", filepath, e)

        logger.info(
            "Cache cleanup completed. Deleted %d files (%sMB). New size: %.2fMB",
            deleted_count, deleted_size / 1024 / 1024, current_size / 1024 / 1024
        )


# Decorator for function caching
def cache_result(
    cache: Optional[FileCache], key_prefix: str = "func", ttl: Optional[int] = None
) -> Callable:
    """Decorator to cache the result of a function."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # If no cache is provided, just execute the function without caching
            if cache is None:
                return func(*args, **kwargs)

            # Generate a unique cache key based on function name, args, and kwargs
            key_parts = [key_prefix, func.__name__] + [str(arg) for arg in args]
            key_parts.extend(sorted([f"{k}={v}" for k, v in kwargs.items()]))
            cache_key = "::".join(key_parts)

            # Try to get the result from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit for key '%s::%s'", key_prefix, func.__name__)
                return cached_result

            logger.debug("Cache miss for key '%s::%s'. Executing function.", key_prefix, func.__name__)

            # Execute the function, cache the result, and handle potential errors
            try:
                result = func(*args, **kwargs)
                cache.set(cache_key, result, ttl=ttl)
                logger.debug("Result for key '%s::%s' cached.", key_prefix, func.__name__)
                return result
            # Catch specific caching errors first
            except (OSError, pickle.PickleError) as cache_exc:
                logger.error(
                    "Error during caching for key '%s::%s': %s",
                    key_prefix, func.__name__, cache_exc
                )
                # Return None as per original logic on caching failure
                return None
            # Catch other exceptions from the function execution
            # Disable W0718 as catching general Exception here is intentional
            # to log errors from the decorated function.
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.error(
                    # Use % formatting for logging
                    # Wrapped long log message
                    "Error during function execution for key '%s::%s': %s",
                    key_prefix, func.__name__, e,
                    exc_info=True # Include traceback for function errors
                )
                # Return None as per original logic on function execution failure
                return None

        return wrapper

    return decorator
