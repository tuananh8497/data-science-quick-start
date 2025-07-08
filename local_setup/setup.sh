#!/bin/bash
PYTHON_VERSION="3.12.6"
echo "Using python version: $PYTHON_VERSION"
# Ref: https://github.com/pyenv/pyenv?tab=readme-ov-file#unixmacos
brew update

if command -v pyenv >/dev/null 2>&1 || command -v pyenv-virtualenv >/dev/null 2>&1; then
    echo "pyenv or pyenv-virtualenv is already installed."
else
    echo "Installing pyenv and pyenv-virtualenv..."
    brew install pyenv pyenv-virtualenv
fi


# Ref: https://github.com/pyenv/pyenv?tab=readme-ov-file#set-up-your-shell-environment-for-pyenv
# echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
# echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
# echo 'eval "$(pyenv init -)"' >> ~/.zshrc
# eval "$(pyenv virtualenv-init -)"
# exec "$SHELL"

if pyenv versions --bare | grep "^${PYTHON_VERSION}$"; then
    echo "Python $PYTHON_VERSION is already installed."
else
    echo "Installing Python $PYTHON_VERSION..."
    pyenv install $PYTHON_VERSION
fi

if pyenv virtualenvs --bare | grep "^data-science-quick-start$"; then
    echo "Virtual environment 'data-science-quick-start' already exists."
else
    echo "Creating virtual environment 'data-science-quick-start'..."
    pyenv virtualenv $PYTHON_VERSION data-science-quick-start-$PYTHON_VERSION
    pyenv local data-science-quick-start-$PYTHON_VERSION
fi

echo "Installing packages..."
pip install -r requirements.txt