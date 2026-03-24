#!/bin/bash

# =============================================================================
# Robo-Advisor API - Start Script
# =============================================================================

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   🚀 Robo-Advisor API - Starting...     ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Aller à la racine du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}📂 Working directory: ${NC}$(pwd)"
echo ""

# Vérifier que src/ existe
if [ ! -d "src" ]; then
    echo -e "${RED}✗ ERROR: src/ directory not found!${NC}"
    echo -e "${RED}  Make sure you're in the project root directory${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Project structure found${NC}"

# Activer virtual environment si existe
VENV_PATHS=(
    "./venv/bin/activate"
    "../venv/bin/activate"
    "../venv_nosql/bin/activate"
    "./venv_nosql/bin/activate"
)

VENV_ACTIVATED=false
for venv_path in "${VENV_PATHS[@]}"; do
    if [ -f "$venv_path" ]; then
        echo -e "${GREEN}✓ Activating virtual environment: ${NC}$venv_path"
        source "$venv_path"
        VENV_ACTIVATED=true
        break
    fi
done

if [ "$VENV_ACTIVATED" = false ]; then
    echo -e "${YELLOW} Warning: No virtual environment found${NC}"
    echo -e "${YELLOW}  Continuing with system Python...${NC}"
fi

# Ajouter projet au PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo -e "${GREEN}✓ PYTHONPATH configured${NC}"

# Vérifier Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: ${NC}$PYTHON_VERSION"

# Vérifier uvicorn installé
if ! command -v uvicorn &> /dev/null; then
    echo -e "${RED}✗ ERROR: uvicorn not found!${NC}"
    echo -e "${RED}  Install with: pip install uvicorn${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Uvicorn found${NC}"

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    API Endpoints Available                ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo -e "${GREEN}   Base:    ${NC}http://localhost:8000"
echo -e "${GREEN}   Swagger: ${NC}http://localhost:8000/docs"
echo -e "${GREEN}   Health:  ${NC}http://localhost:8000/health"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Lancer API
uvicorn src.presentation.api.main:app --reload --host 0.0.0.0 --port 8000
