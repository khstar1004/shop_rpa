#!/usr/bin/env python
"""Script to run all unit tests in the project"""

import unittest
import sys
import os
import argparse

def run_tests(verbosity=1, pattern='test_*.py', start_dir='tests'):
    """Run all tests matching the pattern from the specified directory"""
    # Add parent directory to path for imports to work
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Discover and run tests
    suite = unittest.defaultTestLoader.discover(start_dir, pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return 0 if successful, 1 if there were failures or errors
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the suite of unit tests for the Shop RPA application')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    parser.add_argument('-p', '--pattern', default='test_*.py', help='Pattern to match test files (default: test_*.py)')
    parser.add_argument('-d', '--directory', default='tests', help='Directory to start test discovery (default: tests)')
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    result = run_tests(verbosity=verbosity, pattern=args.pattern, start_dir=args.directory)
    
    sys.exit(result) 