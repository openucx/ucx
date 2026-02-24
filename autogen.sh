#!/bin/sh

usage()
{
	echo "Usage: autogen.sh <options>"
	echo
	echo "  -h|--help                    Show this help message"
	echo "  --with-plugin-repo=URL       Clone plugin repository from URL"
	echo "                               URL format: URL or URL#branch_name"
	echo "                               (default branch: main)"
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
	# Parse branch from URL (format: URL#branch_name), default to "main"
	plugin_branch="main"
	if echo "$plugin_repo_url" | grep -q '#'; then
		plugin_branch="${plugin_repo_url##*#}"
		plugin_repo_url="${plugin_repo_url%#*}"
	fi

	if [ -d "$plugin_dir" ]; then
		# Directory exists
		if [ -d "$plugin_dir/.git" ]; then
			# It's a git repository, pull updates
			echo "Plugin directory exists, pulling latest changes from branch '$plugin_branch'..."
			(cd "$plugin_dir" && git fetch origin && git checkout "$plugin_branch" 2>/dev/null; git pull origin "$plugin_branch") || {
				echo "Error: Failed to pull plugin updates from branch '$plugin_branch'"
				exit 1
			}
		else
			# Directory exists but isn't a git repo
			echo "Warning: $plugin_dir exists but is not a git repository."
			echo "Skipping plugin clone. Remove directory manually to re-clone."
			exit 1
		fi
	else
		# Directory doesn't exist, clone repository
		echo "Cloning plugin repository from $plugin_repo_url (branch: $plugin_branch) to $plugin_dir..."

		# Create parent directories if needed
		parent_dir=$(dirname "$plugin_dir")
		if [ "$parent_dir" != "." ] && [ "$parent_dir" != "$plugin_dir" ]; then
			mkdir -p "$parent_dir" || {
				echo "Error: Failed to create plugin directory: $parent_dir"
				exit 1
			}
		fi

		git clone -b "$plugin_branch" "$plugin_repo_url" "$plugin_dir" || {
			echo "Error: Failed to clone plugin repository from $plugin_repo_url (branch: $plugin_branch)"
			echo "Check URL, branch name, and network connection."
			exit 1
		}
	fi
fi

git submodule update --init src/uct/ib/mlx5/gdaki/gpunetio

autoreconf -v --install || exit 1
rm -rf autom4te.cache
