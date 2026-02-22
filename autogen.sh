#!/bin/sh

usage()
{
	echo "Usage: autogen.sh <options>"
	echo
	echo "  -h|--help                    Show this help message"
	echo "  --with-plugin-repo=URL       Clone plugin repository from URL"
	echo "                               URL format: URL or URL#branch_name"
	echo "                               (default: src/uct/ib/plugin/ucx_plugin/)"
	echo "  --plugin-dir=DIR             Specify custom directory path for plugin"
	echo "                               (default: src/uct/ib/plugin/ucx_plugin/)"
	echo
}

plugin_repo_url=""
plugin_dir=""

for key in "$@"
do
	case $key in
	-h|--help)
		usage
		exit 0
		;;
	--with-plugin-repo=*)
		plugin_repo_url="${key#*=}"
		;;
	--plugin-dir=*)
		plugin_dir="${key#*=}"
		;;
	*)
		# Unknown option - but don't fail, might be for other tools
		# Just continue (autogen.sh is simple and doesn't need strict validation)
		;;
	esac
done

# Set default plugin directory if not specified
if [ -z "$plugin_dir" ]; then
	plugin_dir="src/uct/ib/plugin/ucx_plugin"
fi

rm -rf autom4te.cache
mkdir -p config/m4 config/aux

# Clone or update plugin repository if --with-plugin-repo is provided
# This is completely optional - if not provided, autogen.sh continues normally
if [ -n "$plugin_repo_url" ]; then
	# Parse branch from URL (format: URL#branch_name)
	plugin_branch=""
	if echo "$plugin_repo_url" | grep -q '#'; then
		plugin_branch="${plugin_repo_url##*#}"
		plugin_repo_url="${plugin_repo_url%#*}"
	fi
	
	# Determine full path to plugin directory
	# Relative paths are relative to where autogen.sh is run (source root)
	if [ "${plugin_dir#/}" = "$plugin_dir" ]; then
		# Relative path - use as-is (relative to current directory/source root)
		full_plugin_dir="$plugin_dir"
	else
		# Absolute path - use as-is
		full_plugin_dir="$plugin_dir"
	fi
	
	if [ -d "$full_plugin_dir" ]; then
		# Directory exists
		if [ -d "$full_plugin_dir/.git" ]; then
			# It's a git repository, pull updates
			if [ -n "$plugin_branch" ]; then
				# Use specified branch
				echo "Plugin directory exists, pulling latest changes from branch '$plugin_branch'..."
				(cd "$full_plugin_dir" && git fetch origin && git checkout "$plugin_branch" 2>/dev/null; git pull origin "$plugin_branch") || {
					echo "Error: Failed to pull plugin updates from branch '$plugin_branch'"
					exit 1
				}
			else
				# Use current branch
				echo "Plugin directory exists, pulling latest changes..."
				(cd "$full_plugin_dir" && git pull origin $(git branch --show-current 2>/dev/null || echo "main")) || {
					echo "Error: Failed to pull plugin updates from $plugin_repo_url"
					exit 1
				}
			fi
		else
			# Directory exists but isn't a git repo
			echo "Warning: $full_plugin_dir exists but is not a git repository."
			echo "Skipping plugin clone. Remove directory manually to re-clone."
			exit 1
		fi
	else
		# Directory doesn't exist, clone repository
		if [ -n "$plugin_branch" ]; then
			echo "Cloning plugin repository from $plugin_repo_url (branch: $plugin_branch) to $full_plugin_dir..."
		else
			echo "Cloning plugin repository from $plugin_repo_url to $full_plugin_dir..."
		fi
		# Create parent directories if needed
		parent_dir=$(dirname "$full_plugin_dir")
		if [ "$parent_dir" != "." ] && [ "$parent_dir" != "$full_plugin_dir" ]; then
			mkdir -p "$parent_dir" || {
				echo "Error: Failed to create plugin directory: $parent_dir"
				exit 1
			}
		fi
		if [ -n "$plugin_branch" ]; then
			# Clone specific branch
			git clone -b "$plugin_branch" "$plugin_repo_url" "$full_plugin_dir" || {
				echo "Error: Failed to clone plugin repository from $plugin_repo_url (branch: $plugin_branch)"
				echo "Check URL, branch name, and network connection."
				exit 1
			}
		else
			# Clone default branch
			git clone "$plugin_repo_url" "$full_plugin_dir" || {
				echo "Error: Failed to clone plugin repository from $plugin_repo_url"
				echo "Check URL and network connection."
				exit 1
			}
		fi
	fi
fi

git submodule update --init src/uct/ib/mlx5/gdaki/gpunetio

autoreconf -v --install || exit 1
rm -rf autom4te.cache
