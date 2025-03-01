"""
Dependency Management for MSc-COM7003-AI-Model Project

This script provides functions to check for required packages and install any missing dependencies.
It also includes a list of all required packages with their version specifications.

Usage:
    python dependancies.py

Author: AI Assistant
Created: March 1, 2025
Modified: Current date - Updated to reflect simplified visualization dependencies
"""

import subprocess
import sys
import importlib

# Dictionary of required packages with their minimum versions
REQUIRED_PACKAGES = {
    # Core data processing libraries
    'pandas': '1.3.0',
    'numpy': '1.20.0',
    
    # Visualization libraries
    'matplotlib': '3.4.0',
    
    # Data acquisition libraries
    'yfinance': '0.1.70',
    
    # Optional development libraries
    'jupyter': '1.0.0',
    'pytest': '6.0.0'
}

def check_dependencies():
    """
    Check if all required packages are installed and meet minimum version requirements.
    
    Returns:
        tuple: (all_installed, missing_packages, outdated_packages)
    """
    missing_packages = []
    outdated_packages = []
    
    for package, min_version in REQUIRED_PACKAGES.items():
        try:
            # Try to import the package
            module = importlib.import_module(package)
            
            # Check version (if package has a __version__ attribute)
            if hasattr(module, '__version__'):
                current_version = module.__version__
                
                # Simple version comparison (assumes semantic versioning)
                if current_version.split('.') < min_version.split('.'):
                    outdated_packages.append({
                        'package': package,
                        'current': current_version,
                        'required': min_version
                    })
                    
            print(f"✓ {package} is installed")
            
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is not installed")
    
    all_installed = len(missing_packages) == 0 and len(outdated_packages) == 0
    
    return all_installed, missing_packages, outdated_packages

def install_dependencies(packages=None):
    """
    Install the specified packages using pip.
    
    Args:
        packages (list, optional): List of package names to install. If None, installs all missing packages.
    
    Returns:
        bool: True if all packages were installed successfully, False otherwise.
    """
    if packages is None:
        # Check which packages are missing
        _, missing_packages, _ = check_dependencies()
        packages = missing_packages
    
    if not packages:
        print("No packages to install.")
        return True
    
    print(f"Installing {len(packages)} package(s): {', '.join(packages)}")
    
    # Install each package with its minimum version
    for package in packages:
        if package in REQUIRED_PACKAGES:
            version_spec = REQUIRED_PACKAGES[package]
            package_spec = f"{package}>={version_spec}"
            
            print(f"Installing {package_spec}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_spec])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}")
                return False
    
    return True

def main():
    """Main function to check dependencies and install missing ones if requested."""
    print("Checking dependencies for MSc-COM7003-AI-Model Project...\n")
    
    all_installed, missing_packages, outdated_packages = check_dependencies()
    
    if all_installed:
        print("\nAll dependencies are satisfied!")
        return
    
    # Report missing packages
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
    
    # Report outdated packages
    if outdated_packages:
        print("\nOutdated packages:")
        for pkg in outdated_packages:
            print(f"  {pkg['package']}: {pkg['current']} (required: >={pkg['required']})")
    
    # Ask if the user wants to install missing packages
    if missing_packages:
        answer = input("\nWould you like to install missing packages? (y/n): ")
        if answer.lower() in ['y', 'yes']:
            install_dependencies(missing_packages)
            
    # Ask if the user wants to update outdated packages
    if outdated_packages:
        outdated_names = [pkg['package'] for pkg in outdated_packages]
        answer = input("\nWould you like to update outdated packages? (y/n): ")
        if answer.lower() in ['y', 'yes']:
            install_dependencies(outdated_names)
    
    # Final check
    print("\nFinal dependency check:")
    all_installed, _, _ = check_dependencies()
    
    if all_installed:
        print("\nAll dependencies are now satisfied!")
    else:
        print("\nSome dependencies are still missing or outdated.")
        print("Please check the output above and install them manually if needed.")

if __name__ == "__main__":
    main()
