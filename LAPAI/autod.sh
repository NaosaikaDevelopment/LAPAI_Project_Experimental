#!/bin/bash
cd "$(dirname "$0")"
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_DIR="$SCRIPT_DIR/LAPAI-env"
VENV_PATH="$ENV_DIR/bin/activate"
FISH_VENV_PATH="$ENV_DIR/bin/activate.fish"

if ! command -v python &>/dev/null; then
    echo "Error: python not found"
    exit 1
fi

python -m venv "$ENV_DIR"

echo "Updating pip..."
"$ENV_DIR/bin/python" -m pip install --upgrade pip

echo "Installing requirements..."
"$ENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-l.txt"
"$ENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-strict.txt"


if command -v git-lfs &>/dev/null; then
    git lfs install
else
    echo "Warning: git-lfs not installed"
    echo "Install it if the embedding model downloads only a few MB."
fi

if [ -d "$SCRIPT_DIR/MainCore" ]; then
    cd "$SCRIPT_DIR/MainCore"

    if [ ! -d "multilingual-e5-small" ]; then
        git clone https://huggingface.co/intfloat/multilingual-e5-small
    else
        echo "multilingual-e5-small already exists, skipping clone."
    fi
else
    echo "Warning: MainCore directory not found."
fi

echo
echo "If embedding model downloaded only a few MB, install git-lfs."
echo
echo "Installation done."
echo

SHORTCUT_BLOCK=$(cat <<EOF

# Shortcut for project virtual environment
function nd() {
    if [ -f "$VENV_PATH" ]; then
        source "$VENV_PATH"
        echo "✓ Virtual environment activated successfully!"
    else
        echo "✗ venv file not found at: $VENV_PATH"
        echo "Please run setup.sh again to repair."
    fi
}
EOF
)

echo "Detecting available shells and configuring 'nd' shortcut..."

# Bash
BASH_RC="$HOME/.bashrc"
BASH_PROFILE="$HOME/.bash_profile"

touch "$BASH_RC"

if ! grep -q "function nd()" "$BASH_RC"; then
    echo "$SHORTCUT_BLOCK" >> "$BASH_RC"
    echo "✓ Registered 'nd' shortcut to $BASH_RC"
fi

if [ ! -f "$BASH_PROFILE" ]; then
    cat > "$BASH_PROFILE" <<EOF
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
EOF
fi

# Zsh
ZSH_RC="$HOME/.zshrc"
touch "$ZSH_RC"

if ! grep -q "function nd()" "$ZSH_RC"; then
    echo "$SHORTCUT_BLOCK" >> "$ZSH_RC"
    echo "✓ Registered 'nd' shortcut to $ZSH_RC"
fi

# Fish
if [ -d "$HOME/.config/fish" ] || command -v fish &>/dev/null; then
    FISH_DIR="$HOME/.config/fish/functions"
    mkdir -p "$FISH_DIR"

    cat > "$FISH_DIR/nd.fish" <<EOF
function nd
    if test -f "$FISH_VENV_PATH"
        source "$FISH_VENV_PATH"
        echo "✓ Virtual environment activated successfully!"
    else
        echo "✗ venv file not found at: $FISH_VENV_PATH"
        echo "Please run setup.sh again to repair."
    fi
end
EOF

    echo "✓ Registered 'nd' shortcut for Fish shell"
fi
cd ../
./LAPAI-env/bin/python toreg.py
echo
echo "Setup complete!"
echo "Close this terminal, open a new terminal, then type:"
echo
echo "nd"
echo

read -p "Press Enter to continue..."
