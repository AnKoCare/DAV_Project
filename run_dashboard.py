#!/usr/bin/env python3
"""
Quick script to run the Gaming Behavior Analytics Dashboard
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from dashboard.app import GamingBehaviorDashboard
    from config.config import DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG
    
    print("🎮 Gaming Behavior Analytics Dashboard")
    print("=" * 50)
    print("Starting dashboard...")
    print(f"URL: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    # Create and run dashboard
    dashboard = GamingBehaviorDashboard()
    dashboard.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG)
    
except ImportError as e:
    print(f"Error importing dashboard modules: {e}")
    print("Please make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"Error running dashboard: {e}")
    sys.exit(1) 