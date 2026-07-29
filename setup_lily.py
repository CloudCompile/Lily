#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.5 — VPS Setup Script

Clones, configures, and installs Lily as a non-root user service.
Designed for Ubuntu/Debian VPS. No root required.

Usage:
    python3 setup_lily.py

What it does:
    1. Checks system prerequisites (Python 3.10+, git, pip)
    2. Clones the Lily repo from GitHub
    3. Creates a Python virtual environment
    4. Installs all pip dependencies
    5. Creates the .env file from .env.example (interactive prompts)
    6. Creates the data/ directory
    7. Sets up a systemd user service (auto-start on boot)
    8. Validates the installation
    9. Starts the bot

Run this as a regular (non-root) user on your VPS.
"""

from __future__ import annotations

import os
import sys
import subprocess
import shutil
import json
import textwrap
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────
REPO_URL = "https://github.com/cloudcompile/Lily.git"
REPO_DIR = "Lily"                       # Directory name after clone
BOT_SUBDIR = "Lily"                     # The actual bot code lives inside Lily/Lily/
VENV_DIR = ".venv"                      # Virtual environment directory name
SERVICE_NAME = "lily-bot"              # systemd service name
REQUIREMENTS = "requirements.txt"       # pip requirements file
ENV_EXAMPLE = ".env.example"           # .env template
ENV_FILE = ".env"                       # Actual .env file

# Minimum Python version
MIN_PYTHON = (3, 10)

# ── Colors ─────────────────────────────────────────────────
class Colors:
    """ANSI color codes for pretty terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    PINK    = "\033[95m"

def banner():
    """Print the Lily setup banner."""
    print(f"""
{Colors.PINK}{Colors.BOLD}
    ╔══════════════════════════════════════════╗
    ║     🌸 Lily v8.5 — VPS Setup Script 🌸  ║
    ║         "Lily Lives" Installer            ║
    ╚══════════════════════════════════════════╝
{Colors.RESET}
{Colors.DIM}  This script will set up Lily on your VPS as a non-root user.
  It'll clone the repo, create a virtualenv, install deps, and
  configure a systemd user service so she starts on boot.{Colors.RESET}
""")

def step(msg: str):
    """Print a step header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}── {msg} ──{Colors.RESET}")

def ok(msg: str = "OK"):
    """Print a success message."""
    print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")

def warn(msg: str):
    """Print a warning."""
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {msg}")

def fail(msg: str):
    """Print a failure and exit."""
    print(f"  {Colors.RED}✗{Colors.RESET} {msg}")
    sys.exit(1)

def info(msg: str):
    """Print informational text."""
    print(f"  {Colors.BLUE}ℹ{Colors.RESET} {msg}")

def prompt(msg: str, default: str = "") -> str:
    """Prompt the user for input with a default value."""
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  {Colors.PINK}▸{Colors.RESET} {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    return value if value else default

def prompt_yesno(msg: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer."""
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        value = input(f"  {Colors.PINK}▸{Colors.RESET} {msg}{suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if not value:
        return default
    return value in ("y", "yes", "1", "true")

def run(cmd: list[str] | str, check: bool = True, capture: bool = False, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a shell command with nice output."""
    if isinstance(cmd, str):
        cmd_str = cmd
    else:
        cmd_str = " ".join(str(c) for c in cmd)

    print(f"  {Colors.DIM}$ {cmd_str}{Colors.RESET}")

    result = subprocess.run(
        cmd if isinstance(cmd, str) else cmd,
        shell=isinstance(cmd, str),
        check=False,
        capture_output=capture,
        text=True,
        cwd=str(cwd) if cwd else None,
    )

    if check and result.returncode != 0:
        stderr = result.stderr.strip() if capture else ""
        fail(f"Command failed (exit {result.returncode}): {cmd_str}\n{stderr}")

    return result

def cmd_exists(name: str) -> bool:
    """Check if a command exists on the system."""
    return shutil.which(name) is not None


# ══════════════════════════════════════════════════════════
#  STEP 1: System Prerequisites
# ══════════════════════════════════════════════════════════

def check_prerequisites():
    """Verify system meets minimum requirements."""
    step("Checking system prerequisites")

    # Check Python version
    py_version = sys.version_info
    if py_version < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, found {py_version.major}.{py_version.minor}")
    ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    # Check git
    if not cmd_exists("git"):
        fail("git is not installed. Install it with: sudo apt install git")
    git_result = run(["git", "--version"], capture=True)
    ok(f"git {git_result.stdout.strip()}")

    # Check pip
    pip_result = run([sys.executable, "-m", "pip", "--version"], capture=True, check=False)
    if pip_result.returncode != 0:
        warn("pip not found. Attempting to install via ensurepip...")
        run([sys.executable, "-m", "ensurepip", "--upgrade"], check=False)
        pip_result = run([sys.executable, "-m", "pip", "--version"], capture=True, check=False)
        if pip_result.returncode != 0:
            fail("pip is not available. Install it with: sudo apt install python3-pip")
    ok(f"pip {pip_result.stdout.strip().split()[1]}")

    # Check venv module
    venv_check = run([sys.executable, "-c", "import venv"], capture=True, check=False)
    if venv_check.returncode != 0:
        fail("Python venv module not available. Install it with: sudo apt install python3-venv")
    ok("venv module available")

    # Check systemd (for user services)
    if cmd_exists("systemctl"):
        ok("systemctl available (user service supported)")
    else:
        warn("systemctl not found — won't be able to set up auto-start service")

    # Check we're not root
    if os.geteuid() == 0:
        warn("You're running as root! This script is designed for non-root users.")
        if not prompt_yesno("Continue anyway?", default=False):
            fail("Aborted. Please run as a regular user.")
    else:
        ok(f"Running as non-root user: {os.environ.get('USER', 'unknown')}")


# ══════════════════════════════════════════════════════════
#  STEP 2: Clone the Repository
# ══════════════════════════════════════════════════════════

def clone_repo(install_dir: Path) -> Path:
    """Clone the Lily repository if it doesn't exist already."""
    step("Cloning Lily repository")

    repo_path = install_dir / REPO_DIR

    if repo_path.exists():
        ok(f"Repository already exists at {repo_path}")
        if prompt_yesno("Pull latest changes?", default=True):
            run(["git", "pull"], cwd=repo_path)
            ok("Pulled latest changes")
        return repo_path

    run(["git", "clone", REPO_URL, str(repo_path)])
    ok(f"Cloned to {repo_path}")
    return repo_path


# ══════════════════════════════════════════════════════════
#  STEP 3: Create Virtual Environment
# ══════════════════════════════════════════════════════════

def create_venv(bot_dir: Path) -> Path:
    """Create a Python virtual environment."""
    step("Creating virtual environment")

    venv_path = bot_dir / VENV_DIR

    if venv_path.exists():
        ok(f"Virtual environment already exists at {venv_path}")
        return venv_path

    run([sys.executable, "-m", "venv", str(venv_path)])
    ok(f"Created virtual environment at {venv_path}")

    # Upgrade pip inside the venv
    pip_path = venv_path / "bin" / "pip"
    run([str(pip_path), "install", "--upgrade", "pip"], check=False)
    ok("Upgraded pip in virtual environment")

    return venv_path


# ══════════════════════════════════════════════════════════
#  STEP 4: Install Dependencies
# ══════════════════════════════════════════════════════════

def install_deps(venv_path: Path, bot_dir: Path):
    """Install Python dependencies from requirements.txt."""
    step("Installing Python dependencies")

    pip_path = venv_path / "bin" / "pip"
    req_file = bot_dir / REQUIREMENTS

    if not req_file.exists():
        fail(f"requirements.txt not found at {req_file}")

    run([str(pip_path), "install", "-r", str(req_file)])
    ok("All dependencies installed")

    # Verify critical packages
    python_path = venv_path / "bin" / "python"
    for pkg in ["discord", "aiohttp", "dotenv"]:
        result = run(
            [str(python_path), "-c", f"import {pkg}"],
            capture=True, check=False
        )
        if result.returncode != 0:
            warn(f"Package '{pkg}' may not be installed correctly")
        else:
            ok(f"Verified: {pkg}")


# ══════════════════════════════════════════════════════════
#  STEP 5: Create .env Configuration
# ══════════════════════════════════════════════════════════

def create_env(bot_dir: Path, pollinations_key: str = ""):
    """Create the .env file from .env.example with user input."""
    step("Configuring .env file")

    env_path = bot_dir / ENV_FILE
    example_path = bot_dir / ENV_EXAMPLE

    if env_path.exists():
        ok(f".env already exists at {env_path}")
        if not prompt_yesno("Overwrite with new configuration?", default=False):
            info("Keeping existing .env file")
            return
        # Backup existing
        backup = env_path.with_suffix(f".env.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(env_path, backup)
        ok(f"Backed up existing .env to {backup}")

    if not example_path.exists():
        fail(f".env.example not found at {example_path}")

    # Read the example
    example_content = example_path.read_text()

    # Interactive configuration
    print(f"\n  {Colors.BOLD}Let's configure Lily!{Colors.RESET}")
    print(f"  {Colors.DIM}Press Enter to accept defaults.{Colors.RESET}\n")

    # Discord token (REQUIRED)
    discord_token = prompt("Discord Bot Token (REQUIRED)", default="")
    if not discord_token:
        warn("No Discord token provided — you'll need to add it to .env manually!")

    # Pollinations key
    pollinations = prompt("Pollinations API Key", default=pollinations_key or "")

    # Admin IDs
    admin_ids = prompt("Admin Discord User IDs (comma-separated)", default="")

    # Bot prefix
    bot_prefix = prompt("Bot command prefix", default="!lily")

    # v8.5 Features
    print(f"\n  {Colors.BOLD}v8.5 Features:{Colors.RESET}")
    proactive = prompt_yesno("Enable proactive DMs?", default=True)
    recaps = prompt_yesno("Enable daily recaps?", default=True)
    dreams = prompt_yesno("Enable dream journal?", default=True)
    mood = prompt_yesno("Enable mood-reactive status?", default=True)

    # Default models
    text_model = prompt("Default text model", default="openai-fast")
    image_model = prompt("Default image model", default="sana")

    # Build the .env content
    env_content = f"""# Lily v8.5 Configuration
# Generated by setup_lily.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# ── Discord ──────────────────────────────────────────────
DISCORD_TOKEN={discord_token}

# ── Pollinations API ─────────────────────────────────────
POLLINATIONS_KEY={pollinations}
POLLINATIONS_BASE_URL=https://gen.pollinations.ai
POLLINATIONS_MEDIA_URL=https://media.pollinations.ai

# ── Admin IDs (comma-separated) ──────────────────────────
ADMIN_IDS={admin_ids}

# ── Bot behaviour ────────────────────────────────────────
BOT_PREFIX={bot_prefix}

# ── Default models (v8.5: cheap models by default!) ──────
DEFAULT_TEXT_MODEL={text_model}
DEFAULT_IMAGE_MODEL={image_model}

# ── v8.5 Features ────────────────────────────────────────
PROACTIVE_DM_ENABLED={'true' if proactive else 'false'}
PROACTIVE_DM_CHECK_INTERVAL=300

DAILY_RECAP_ENABLED={'true' if recaps else 'false'}
DAILY_RECAP_HOUR=23

DREAM_JOURNAL_ENABLED={'true' if dreams else 'false'}
DREAM_JOURNAL_HOUR=3

MOOD_STATUS_ENABLED={'true' if mood else 'false'}
MOOD_STATUS_INTERVAL=300
"""

    env_path.write_text(env_content)
    ok(f"Created .env at {env_path}")

    # Warn about missing token
    if not discord_token:
        warn("⚠️  DISCORD_TOKEN is empty! Lily won't start without it.")
        warn("   Edit .env later and add your bot token.")


# ══════════════════════════════════════════════════════════
#  STEP 6: Create Data Directory
# ══════════════════════════════════════════════════════════

def create_data_dir(bot_dir: Path):
    """Create the data directory for the SQLite database."""
    step("Creating data directory")

    data_dir = bot_dir / "data"
    data_dir.mkdir(exist_ok=True)
    ok(f"Data directory ready at {data_dir}")

    # Create a .gitkeep so the dir is tracked
    gitkeep = data_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("")
        ok("Created .gitkeep")


# ══════════════════════════════════════════════════════════
#  STEP 7: Create Systemd User Service
# ══════════════════════════════════════════════════════════

def create_systemd_service(venv_path: Path, bot_dir: Path):
    """Create a systemd user service for Lily (auto-start on boot)."""
    step("Setting up systemd user service")

    if not cmd_exists("systemctl"):
        warn("systemctl not found — skipping service setup")
        warn("You can start Lily manually with: cd {bot_dir} && .venv/bin/python bot.py")
        return

    python_path = venv_path / "bin" / "python"
    bot_script = bot_dir / "bot.py"

    # Ensure systemd user directory exists
    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)

    service_content = f"""[Unit]
Description=Lily v8.5 Discord Bot — "Lily Lives"
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={bot_dir}
ExecStart={python_path} {bot_script}
Restart=on-failure
RestartSec=15
StartLimitBurst=5
StartLimitIntervalSec=300

# Environment
Environment=PYTHONUNBUFFERED=1
Environment=LANG=en_US.UTF-8

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=lily-bot

# Resource limits (gentle)
MemoryMax=512M
CPUQuota=80%

[Install]
WantedBy=default.target
"""

    service_file = systemd_dir / f"{SERVICE_NAME}.service"
    service_file.write_text(service_content)
    ok(f"Created service file at {service_file}")

    # Reload systemd user daemon
    run(["systemctl", "--user", "daemon-reload"], check=False)
    ok("Reloaded systemd user daemon")

    # Enable lingering (so service runs even when not logged in)
    if cmd_exists("loginctl"):
        run(["loginctl", "enable-linger", os.environ.get("USER", "")], check=False)
        ok("Enabled login linger (bot runs even when you're logged out)")

    # Ask if they want to enable auto-start
    if prompt_yesno("Enable Lily to start on boot?", default=True):
        run(["systemctl", "--user", "enable", SERVICE_NAME], check=False)
        ok(f"Enabled {SERVICE_NAME} service (auto-start on boot)")
    else:
        info("Auto-start not enabled. You can enable later with:")
        info("  systemctl --user enable lily-bot")


# ══════════════════════════════════════════════════════════
#  STEP 8: Validate Installation
# ══════════════════════════════════════════════════════════

def validate_installation(venv_path: Path, bot_dir: Path):
    """Validate that the installation looks correct."""
    step("Validating installation")

    python_path = venv_path / "bin" / "python"
    checks_passed = 0
    checks_total = 0

    # Check 1: Virtual environment exists
    checks_total += 1
    if venv_path.exists():
        ok("Virtual environment exists")
        checks_passed += 1
    else:
        warn("Virtual environment not found")

    # Check 2: .env file exists
    checks_total += 1
    env_path = bot_dir / ENV_FILE
    if env_path.exists():
        ok(".env file exists")
        checks_passed += 1
    else:
        warn(".env file not found")

    # Check 3: Discord token is set
    checks_total += 1
    if env_path.exists():
        content = env_path.read_text()
        if "DISCORD_TOKEN=" in content and "YOUR_DISCORD_BOT_TOKEN_HERE" not in content:
            # Check it's not empty
            for line in content.splitlines():
                if line.startswith("DISCORD_TOKEN="):
                    token_val = line.split("=", 1)[1].strip()
                    if token_val:
                        ok("Discord token is configured")
                        checks_passed += 1
                    else:
                        warn("Discord token is empty in .env")
                    break
        else:
            warn("Discord token not set in .env")
    else:
        warn("Cannot check Discord token (.env missing)")

    # Check 4: Pollinations key is set
    checks_total += 1
    if env_path.exists():
        content = env_path.read_text()
        for line in content.splitlines():
            if line.startswith("POLLINATIONS_KEY="):
                key_val = line.split("=", 1)[1].strip()
                if key_val:
                    ok("Pollinations API key is configured")
                    checks_passed += 1
                else:
                    warn("Pollinations API key is empty (optional, but recommended)")
                    checks_passed += 1  # Not critical
                break

    # Check 5: Can import discord.py
    checks_total += 1
    result = run(
        [str(python_path), "-c", "import discord; print(discord.__version__)"],
        capture=True, check=False
    )
    if result.returncode == 0:
        ok(f"discord.py {result.stdout.strip()} installed")
        checks_passed += 1
    else:
        warn("discord.py not importable")

    # Check 6: Can import all project modules
    checks_total += 1
    result = run(
        [str(python_path), "-c", "import config; import database; import pollinations; import personality; import relationships; import memories; import model_router; import quotas; print('All modules OK')"],
        capture=True, check=False, cwd=bot_dir
    )
    if result.returncode == 0:
        ok("All Lily modules importable")
        checks_passed += 1
    else:
        warn(f"Some Lily modules failed to import: {result.stderr.strip()[:200]}")

    # Check 7: Data directory exists
    checks_total += 1
    data_dir = bot_dir / "data"
    if data_dir.exists():
        ok("Data directory exists")
        checks_passed += 1
    else:
        warn("Data directory not found")

    # Check 8: bot.py exists
    checks_total += 1
    bot_script = bot_dir / "bot.py"
    if bot_script.exists():
        ok("bot.py exists")
        checks_passed += 1
    else:
        warn("bot.py not found")

    # Summary
    print(f"\n  {Colors.BOLD}Validation: {checks_passed}/{checks_total} checks passed{Colors.RESET}")
    if checks_passed == checks_total:
        ok("All checks passed! Lily is ready to go! 🌸")
    elif checks_passed >= checks_total - 2:
        warn("Most checks passed. Review the warnings above.")
    else:
        warn("Several checks failed. Review the output above before starting Lily.")


# ══════════════════════════════════════════════════════════
#  STEP 9: Start the Bot
# ══════════════════════════════════════════════════════════

def start_bot(venv_path: Path, bot_dir: Path):
    """Start Lily either via systemd or directly."""
    step("Starting Lily")

    python_path = venv_path / "bin" / "python"
    bot_script = bot_dir / "bot.py"

    # Check if token is set
    env_path = bot_dir / ENV_FILE
    if env_path.exists():
        content = env_path.read_text()
        token_set = False
        for line in content.splitlines():
            if line.startswith("DISCORD_TOKEN="):
                token_val = line.split("=", 1)[1].strip()
                if token_val and token_val != "YOUR_DISCORD_BOT_TOKEN_HERE":
                    token_set = True
                break

        if not token_set:
            warn("Discord token is not set in .env!")
            warn("Lily won't start without it. Add your token and run:")
            print(f"    {Colors.BOLD}cd {bot_dir} && {python_path} bot.py{Colors.RESET}")
            print(f"  or:")
            print(f"    {Colors.BOLD}systemctl --user start lily-bot{Colors.RESET}")
            return

    if cmd_exists("systemctl"):
        # Check if the service file exists
        service_file = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
        if service_file.exists():
            if prompt_yesno("Start Lily via systemd service?", default=True):
                run(["systemctl", "--user", "start", SERVICE_NAME], check=False)
                ok(f"Started {SERVICE_NAME} service")
                run(["systemctl", "--user", "status", SERVICE_NAME], check=False)
                return

    # Fallback: offer to start directly
    if prompt_yesno("Start Lily now? (will run in foreground)", default=False):
        print(f"\n  {Colors.PINK}{Colors.BOLD}🌸 Starting Lily...{Colors.RESET}")
        print(f"  {Colors.DIM}Press Ctrl+C to stop.{Colors.RESET}\n")
        os.chdir(bot_dir)
        os.execv(str(python_path), [str(python_path), str(bot_script)])
    else:
        print(f"\n  To start Lily manually:")
        print(f"    {Colors.BOLD}cd {bot_dir} && {python_path} bot.py{Colors.RESET}")
        print(f"\n  Or via systemd (if configured):")
        print(f"    {Colors.BOLD}systemctl --user start lily-bot{Colors.RESET}")
        print(f"\n  To check Lily's status:")
        print(f"    {Colors.BOLD}systemctl --user status lily-bot{Colors.RESET}")
        print(f"\n  To view Lily's logs:")
        print(f"    {Colors.BOLD}journalctl --user -u lily-bot -f{Colors.RESET}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    banner()

    # ── Determine install directory ───────────────────────
    home = Path.home()
    default_install = home

    print(f"  {Colors.BOLD}Where should Lily be installed?{Colors.RESET}")
    install_dir = Path(prompt("Install directory", default=str(default_install)))
    install_dir.mkdir(parents=True, exist_ok=True)

    # Optional: pass Pollinations key via command line
    pollinations_key = ""
    if len(sys.argv) > 1:
        pollinations_key = sys.argv[1]

    # ── Run all steps ─────────────────────────────────────
    check_prerequisites()
    repo_path = clone_repo(install_dir)

    # The actual bot code is in repo_path/Lily/ (nested Lily/Lily/)
    bot_dir = repo_path / BOT_SUBDIR
    if not bot_dir.exists():
        fail(f"Bot directory not found at {bot_dir}. Repository structure may have changed.")

    venv_path = create_venv(bot_dir)
    install_deps(venv_path, bot_dir)
    create_env(bot_dir, pollinations_key=pollinations_key)
    create_data_dir(bot_dir)
    create_systemd_service(venv_path, bot_dir)
    validate_installation(venv_path, bot_dir)
    start_bot(venv_path, bot_dir)

    # ── Final summary ─────────────────────────────────────
    print(f"""
{Colors.PINK}{Colors.BOLD}
    ╔══════════════════════════════════════════╗
    ║     🌸 Lily v8.5 Setup Complete! 🌸     ║
    ╚══════════════════════════════════════════╝
{Colors.RESET}
  {Colors.BOLD}Lily is installed at:{Colors.RESET} {bot_dir}
  {Colors.BOLD}Virtual env:{Colors.RESET}          {venv_path}
  {Colors.BOLD}Config file:{Colors.RESET}          {bot_dir / '.env'}
  {Colors.BOLD}Database:{Colors.RESET}             {bot_dir / 'data' / 'lily.db'} (auto-created on first run)

  {Colors.BOLD}Quick commands:{Colors.RESET}
    Start Lily:      {venv_path / 'bin' / 'python'} {bot_dir / 'bot.py'}
    Service start:   systemctl --user start lily-bot
    Service status:  systemctl --user status lily-bot
    View logs:       journalctl --user -u lily-bot -f
    Stop Lily:       systemctl --user stop lily-bot
    Restart Lily:    systemctl --user restart lily-bot
    Enable auto:     systemctl --user enable lily-bot
    Disable auto:    systemctl --user disable lily-bot

  {Colors.DIM}Lily v8.5 — She lives 💕 | Cross-server memories ✨ | Sana Sprint images 🖼️{Colors.RESET}
""")


if __name__ == "__main__":
    main()
