import os
import shutil

# Create static/images directory if it doesn't exist
os.makedirs('static/images', exist_ok=True)

# Source logo path (your original file)
logo_src = r"C:\Users\Somaa\Downloads\DALL·E 2025-02-05 14.00.31 - A sleek and modern logo for 'Med Mentors', an AI-powered healthcare platform. The logo should feature a futuristic AI chatbot infused with a medical c.webp"

# Destination paths
logo_dst = "static/images/medmentors-logo.webp"

# Copy logo
try:
    shutil.copy2(logo_src, logo_dst)
    print(f"Logo copied successfully to {logo_dst}")
except Exception as e:
    print(f"Error copying logo: {e}") 