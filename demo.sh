#!/bin/bash
#
# Demo script for marimo WebAssembly + GitHub Pages Template
# This script demonstrates how to build the site locally for testing purposes.
# It installs the necessary tools and runs the build script with appropriate parameters.

# Install uv/uvx - a fast, user-friendly Python package installer and resolver
# uv is similar to pip but with improved performance and dependency resolution
# The -LsSf flags for curl mean:
#   -L: Follow redirects
#   -s: Silent mode
#   -S: Show error if fails
#   -f: Fail silently on server errors
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the build.py script using uv
# This is interesting as it works without explicitly installing
# the dependencies before running this script - uv handles this automatically
# 
# Parameters:
#   --output_dir '_site': Specifies where to output the generated site files
#                         '_site' is a common convention for static site generators
#
#   --template 'templates/tailwind.html.j2': Specifies which template to use
#                                            This uses the Tailwind CSS template
#                                            for a clean, responsive design
uv run build.py \
       --output_dir '_site' \
       --template 'templates/tailwind.html.j2'
