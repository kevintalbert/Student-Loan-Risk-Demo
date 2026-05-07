#!/usr/bin/env python3
"""
CML Environment Variables Helper

This script helps you understand the CML environment variables used for
automated model deployment. In most cases, these are automatically available
in CML workspace terminals.
"""

import os
from pathlib import Path

def check_cml_environment():
    """Check if CML environment variables are available."""
    
    print("🔧 CML Environment Variables Check")
    print("="*50)
    
    # Check environment variables
    env_vars = {
        "CDSW_API_URL": "CML workspace API URL",
        "CDSW_APIV2_KEY": "Your CML API key", 
        "CDSW_PROJECT_ID": "Current project ID"
    }
    
    print("\nChecking CML environment variables:")
    all_present = True
    
    for var_name, description in env_vars.items():
        value = os.getenv(var_name)
        if value:
            # Mask sensitive values
            display_value = "***" + value[-4:] if "KEY" in var_name else value
            print(f"  ✅ {var_name}: {display_value}")
        else:
            print(f"  ❌ {var_name}: Not set")
            all_present = False
    
    if all_present:
        print(f"\n🎉 All environment variables are set!")
        print(f"   You can run: python create_model.py")
        return True
    else:
        print(f"\n⚠️  Some environment variables are missing.")
        print(f"   Make sure you're in a CML workspace terminal.")
        return False

def create_manual_config():
    """Create a manual configuration for non-CML environments."""
    
    print("\n" + "="*50)
    print("Manual Configuration (for testing outside CML)")
    print("="*50)
    
    # Get CML workspace information  
    print("\nIf you need to set these manually:")
    print("1. CML Workspace URL (from your CML workspace)")
    print("2. API Key (CML → User Settings → API Keys)")
    print("3. Project ID (from project URL or settings)")
    
    create_config = input("\nCreate manual config file? (y/N): ").strip().lower()
    if create_config == 'y':
        cml_host = input("   Enter CML Host URL: ").strip()
        api_key = input("   Enter API Key: ").strip()
        project_id = input("   Enter Project ID: ").strip()
        
        # Create config file
        config_content = f'''#!/usr/bin/env python3
"""
Manual CML Environment Variables

Set these environment variables to run create_model.py outside of CML.
"""

import os

# Set CML environment variables manually
os.environ["CDSW_API_URL"] = "{cml_host}/api/v1"
os.environ["CDSW_APIV2_KEY"] = "{api_key}"
os.environ["CDSW_PROJECT_ID"] = "{project_id}"

print("Environment variables set. You can now run:")
print("python create_model.py")
'''
        
        # Write config file
        config_file = Path("set_cml_env.py")
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"\n✅ Configuration saved to: {config_file}")
        print("\n🚀 Usage:")
        print("1. python set_cml_env.py")
        print("2. python create_model.py")
        print("\n⚠️  Security Note:")
        print("   - Add set_cml_env.py to .gitignore to protect API keys")
        
        # Update .gitignore
        gitignore_path = Path(".gitignore")
        if gitignore_path.exists():
            with open(gitignore_path, 'a') as f:
                f.write("\n# Manual CML environment config\n")
                f.write("set_cml_env.py\n")
            print("   - Added set_cml_env.py to .gitignore")


if __name__ == "__main__":
    env_check_passed = check_cml_environment()
    
    if not env_check_passed:
        create_manual_config()
    else:
        print(f"\n🚀 Ready to deploy! Run:")
        print(f"   python create_model.py")
