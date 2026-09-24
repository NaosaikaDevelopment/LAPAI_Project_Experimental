#!/bin/bash

echo " Removing 'nd' shortcut from all shell configurations..."

# 1. Clean Bash (.bashrc)
BASH_RC="$HOME/.bashrc"
if [ -f "$BASH_RC" ]; then
    perl -i -0777 -pe 's/# Shortcut for project virtual environment\nfunction nd\(\) \{.*?\n\}\n//s' "$BASH_RC"
    echo "✓ Removed 'nd' from $BASH_RC"
fi

# 2. Clean Zsh (.zshrc)
ZSH_RC="$HOME/.zshrc"
if [ -f "$ZSH_RC" ]; then
    perl -i -0777 -pe 's/# Shortcut for project virtual environment\nfunction nd\(\) \{.*?\n\}\n//s' "$ZSH_RC"
    echo "✓ Removed 'nd' from $ZSH_RC"
fi

# 3. Clean Fish Shell
FISH_FILE="$HOME/.config/fish/functions/nd.fish"
if [ -f "$FISH_FILE" ]; then
    rm "$FISH_FILE"
    echo "✓ Removed 'nd' from Fish functions"
fi

echo " Uninstallation complete! Please restart your terminal window."
#!/bin/bash

echo " Removing 'nd' shortcut from all shell configurations..."

# 1. Clean Bash (.bashrc)
BASH_RC="$HOME/.bashrc"
if [ -f "$BASH_RC" ]; then
    perl -i -0777 -pe 's/# Shortcut for project virtual environment\nfunction nd\(\) \{.*?\n\}\n//s' "$BASH_RC"
    echo "✓ Removed 'nd' from $BASH_RC"
fi

# 2. Clean Zsh (.zshrc)
ZSH_RC="$HOME/.zshrc"
if [ -f "$ZSH_RC" ]; then
    perl -i -0777 -pe 's/# Shortcut for project virtual environment\nfunction nd\(\) \{.*?\n\}\n//s' "$ZSH_RC"
    echo "✓ Removed 'nd' from $ZSH_RC"
fi

# 3. Clean Fish Shell
FISH_FILE="$HOME/.config/fish/functions/nd.fish"
if [ -f "$FISH_FILE" ]; then
    rm "$FISH_FILE"
    echo "✓ Removed 'nd' from Fish functions"
fi

echo "💡 Uninstallation complete! Please restart your terminal window."
