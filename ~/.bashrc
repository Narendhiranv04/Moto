
# Auto-activate .venv in Moto project
if [ -d "/home/naren/Moto/.venv" ] && [[ "$PWD" == "/home/naren/Moto"* ]]; then
    source "/home/naren/Moto/.venv/bin/activate"
fi
