#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌸 Lily + NullVector — Unified VPS Setup Script

Clones, configures, and installs BOTH Lily and NullVector on a non-root VPS.
Creates virtual environments, installs deps, sets up systemd user services,
and validates everything.

Usage:
    python3 setup_bots.py

What it does:
    1. Checks system prerequisites (Python 3.10+, git, pip, venv)
    2. Clones both repos from GitHub
    3. Creates separate virtual environments for each bot
    4. Installs all pip dependencies
    5. Creates .env files from .env.example (interactive prompts)
    6. Creates data/ directories
    7. Sets up systemd user services (auto-start on boot)
    8. Validates both installations
    9. Starts the bots

Run this as a regular (non-root) user on your VPS.
"""

from __future__ import annotations

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# ── Configuration ──────────────────────────────────────────

@dataclass
class BotConfig:
    """Configuration for a single bot."""
    name: str
    repo_url: str
    repo_dir: str
    bot_subdir: str
    service_name: str
    description: str
    color: str

BOTS = [
    BotConfig(
        name="Lily",
        repo_url="https://github.com/cloudcompile/Lily.git",
        repo_dir="Lily",
        bot_subdir="Lily",
        service_name="lily-bot",
        description="AI Discord Bot who actually feels real",
        color="\033[95m",  # Pink
    ),
    BotConfig(
        name="NullVector",
        repo_url="https://github.com/cloudcompile/nullvector.git",
        repo_dir="nullvector",
        bot_subdir=".",
        service_name="nullvector-bot",
        description="Smart AI assistant with intelligent model routing",
        color="\033[94m",  # Blue
    ),
]

VENV_DIR = ".venv"
MIN_PYTHON = (3, 10)

# ── Colors ─────────────────────────────────────────────────

class C:
    """ANSI color codes."""
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
    """Print the setup banner."""
    print(f"""
{C.PINK}{C.BOLD}
    ╔═══════════════════════════════════════════════╗
    ║  🌸 Lily + NullVector — Unified VPS Setup 🌸  ║
    ║       Dual Bot Installer for Non-Root VPS     ║
    ╚═══════════════════════════════════════════════╝
{C.RESET}
{C.DIM}  This script will set up BOTH bots on your VPS as a non-root user.
  Each bot gets its own virtualenv, .env, and systemd service.
  They'll auto-start on boot and restart on crashes.{C.RESET}
""")


def step(msg: str, bot_name: str = ""):
    """Print a step header."""
    prefix = f"[{bot_name}] " if bot_name else ""
    print(f"\n{C.CYAN}{C.BOLD}── {prefix}{msg} ──{C.RESET}")


def ok(msg: str = "OK"):
    print(f"  {C.GREEN}✓{C.RESET} {msg}")


def warn(msg: str):
    print(f"  {C.YELLOW}⚠{C.RESET} {msg}")


def fail(msg: str):
    print(f"  {C.RED}✗{C.RESET} {msg}")
    sys.exit(1)


def info(msg: str):
    print(f"  {C.BLUE}ℹ{C.RESET} {msg}")


def prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  {C.PINK}▸{C.RESET} {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    return value if value else default


def prompt_yesno(msg: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        value = input(f"  {C.PINK}▸{C.RESET} {msg}{suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if not value:
        return default
    return value in ("y", "yes", "1", "true")


def run(cmd: list | str, check: bool = True, capture: bool = False, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a shell command."""
    if isinstance(cmd, str):
        cmd_str = cmd
    else:
        cmd_str = " ".join(str(c) for c in cmd)

    print(f"  {C.DIM}$ {cmd_str}{C.RESET}")

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
    return shutil.which(name) is not None


# ══════════════════════════════════════════════════════════
#  STEP 1: System Prerequisites
# ══════════════════════════════════════════════════════════

def check_prerequisites():
    """Verify system meets minimum requirements."""
    step("Checking system prerequisites")

    py_version = sys.version_info
    if py_version < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, found {py_version.major}.{py_version.minor}")
    ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    if not cmd_exists("git"):
        fail("git is not installed. Install it with: sudo apt install git")
    ok("git available")

    pip_result = run([sys.executable, "-m", "pip", "--version"], capture=True, check=False)
    if pip_result.returncode != 0:
        fail("pip not found. Install it with: sudo apt install python3-pip")
    ok("pip available")

    venv_check = run([sys.executable, "-c", "import venv"], capture=True, check=False)
    if venv_check.returncode != 0:
        fail("Python venv module not available. Install it with: sudo apt install python3-venv")
    ok("venv module available")

    if cmd_exists("systemctl"):
        ok("systemctl available (user service supported)")
    else:
        warn("systemctl not found — won't be able to set up auto-start service")

    if os.geteuid() == 0:
        warn("You're running as root! This script is designed for non-root users.")
        if not prompt_yesno("Continue anyway?", default=False):
            fail("Aborted. Please run as a regular user.")
    else:
        ok(f"Running as non-root user: {os.environ.get('USER', 'unknown')}")


# ══════════════════════════════════════════════════════════
#  STEP 2: Shared Configuration
# ══════════════════════════════════════════════════════════

def collect_shared_config() -> dict:
    """Collect shared configuration values that apply to both bots."""
    step("Shared configuration")

    print(f"\n  {C.BOLD}Some settings are shared between both bots.{C.RESET}")
    print(f"  {C.DIM}You'll configure bot-specific settings next.{C.RESET}\n")

    # Discord tokens
    lily_token = prompt("Lily Discord Bot Token (REQUIRED)", default="")
    nullvector_token = prompt("NullVector Discord Bot Token (REQUIRED)", default="")

    # Pollinations key (shared)
    pollinations_key = prompt("Pollinations API Key (shared by both bots)", default="")

    # Admin IDs (shared)
    admin_ids = prompt("Admin Discord User IDs (comma-separated, shared)", default="")

    # Install directory
    home = Path.home()
    install_dir = Path(prompt("Install directory", default=str(home)))

    return {
        "lily_token": lily_token,
        "nullvector_token": nullvector_token,
        "pollinations_key": pollinations_key,
        "admin_ids": admin_ids,
        "install_dir": install_dir,
    }


# ══════════════════════════════════════════════════════════
#  STEP 3: Clone Repositories
# ══════════════════════════════════════════════════════════

def clone_repo(bot: BotConfig, install_dir: Path) -> Path:
    """Clone a bot repository if it doesn't exist."""
    step("Cloning repository", bot.name)

    repo_path = install_dir / bot.repo_dir

    if repo_path.exists():
        ok(f"Repository already exists at {repo_path}")
        if prompt_yesno(f"Pull latest changes for {bot.name}?", default=True):
            run(["git", "pull"], cwd=repo_path)
            ok("Pulled latest changes")
        return repo_path

    run(["git", "clone", bot.repo_url, str(repo_path)])
    ok(f"Cloned to {repo_path}")
    return repo_path


# ══════════════════════════════════════════════════════════
#  STEP 4: Create Virtual Environments
# ══════════════════════════════════════════════════════════

def create_venv(bot_dir: Path, bot_name: str) -> Path:
    """Create a Python virtual environment for a bot."""
    step("Creating virtual environment", bot_name)

    venv_path = bot_dir / VENV_DIR

    if venv_path.exists():
        ok(f"Virtual environment already exists at {venv_path}")
        return venv_path

    run([sys.executable, "-m", "venv", str(venv_path)])
    ok(f"Created virtual environment at {venv_path}")

    pip_path = venv_path / "bin" / "pip"
    run([str(pip_path), "install", "--upgrade", "pip"], check=False)
    ok("Upgraded pip in virtual environment")

    return venv_path


# ══════════════════════════════════════════════════════════
#  STEP 5: Install Dependencies
# ══════════════════════════════════════════════════════════

def install_deps(venv_path: Path, bot_dir: Path, bot_name: str):
    """Install Python dependencies for a bot."""
    step("Installing Python dependencies", bot_name)

    pip_path = venv_path / "bin" / "pip"
    req_file = bot_dir / "requirements.txt"

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
#  STEP 6: Create .env Configuration
# ══════════════════════════════════════════════════════════

def create_env(bot_dir: Path, bot_name: str, shared: dict, bot: BotConfig):
    """Create the .env file for a bot."""
    step("Configuring .env", bot_name)

    env_path = bot_dir / ".env"
    example_path = bot_dir / ".env.example"

    if env_path.exists():
        ok(f".env already exists at {env_path}")
        if not prompt_yesno(f"Overwrite {bot_name} .env?", default=False):
            info("Keeping existing .env file")
            return
        backup = env_path.with_suffix(f".env.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(env_path, backup)
        ok(f"Backed up existing .env to {backup}")

    # Determine which token to use
    if bot_name == "Lily":
        discord_token = shared["lily_token"]
    else:
        discord_token = shared["nullvector_token"]

    # Read the example file if it exists
    if example_path.exists():
        content = example_path.read_text()
        # Replace placeholder values
        content = content.replace("YOUR_DISCORD_BOT_TOKEN_HERE", discord_token)
        # Replace empty POLLINATIONS_KEY with the actual key
        lines = content.splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("POLLINATIONS_KEY="):
                line = f"POLLINATIONS_KEY={shared['pollinations_key']}"
            elif line.startswith("DISCORD_TOKEN="):
                line = f"DISCORD_TOKEN={discord_token}"
            elif line.startswith("ADMIN_IDS="):
                line = f"ADMIN_IDS={shared['admin_ids']}"
            new_lines.append(line)
        content = "\n".join(new_lines)

        # Bot-specific prompts
        if bot_name == "Lily":
            print(f"\n  {C.BOLD}Lily v8.5 Features:{C.RESET}")
            proactive = prompt_yesno("  Enable proactive DMs?", default=True)
            recaps = prompt_yesno("  Enable daily recaps?", default=True)
            dreams = prompt_yesno("  Enable dream journal?", default=True)
            mood = prompt_yesno("  Enable mood-reactive status?", default=True)

            # Update feature flags in the content
            new_lines = []
            for line in content.splitlines():
                if line.startswith("PROACTIVE_DM_ENABLED="):
                    line = f"PROACTIVE_DM_ENABLED={'true' if proactive else 'false'}"
                elif line.startswith("DAILY_RECAP_ENABLED="):
                    line = f"DAILY_RECAP_ENABLED={'true' if recaps else 'false'}"
                elif line.startswith("DREAM_JOURNAL_ENABLED="):
                    line = f"DREAM_JOURNAL_ENABLED={'true' if dreams else 'false'}"
                elif line.startswith("MOOD_STATUS_ENABLED="):
                    line = f"MOOD_STATUS_ENABLED={'true' if mood else 'false'}"
                new_lines.append(line)
            content = "\n".join(new_lines)

        elif bot_name == "NullVector":
            print(f"\n  {C.BOLD}NullVector v2.0 Settings:{C.RESET}")
            rate_hourly = prompt("  Hourly rate limit per user", default="30")
            rate_daily = prompt("  Daily rate limit per user", default="100")

            new_lines = []
            for line in content.splitlines():
                if line.startswith("RATE_LIMIT_HOURLY="):
                    line = f"RATE_LIMIT_HOURLY={rate_hourly}"
                elif line.startswith("RATE_LIMIT_DAILY="):
                    line = f"RATE_LIMIT_DAILY={rate_daily}"
                new_lines.append(line)
            content = "\n".join(new_lines)

        env_path.write_text(content)
    else:
        # No .env.example — create from scratch
        content = f"""# {bot_name} Configuration
# Generated by setup_bots.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DISCORD_TOKEN={discord_token}
POLLINATIONS_KEY={shared['pollinations_key']}
POLLINATIONS_BASE_URL=https://gen.pollinations.ai
POLLINATIONS_MEDIA_URL=https://media.pollinations.ai
ADMIN_IDS={shared['admin_ids']}
BOT_PREFIX={'!lily' if bot_name == 'Lily' else '!'}
DEFAULT_TEXT_MODEL=openai-fast
DEFAULT_IMAGE_MODEL=sana
"""
        if bot_name == "Lily":
            content += """
PROACTIVE_DM_ENABLED=true
PROACTIVE_DM_CHECK_INTERVAL=300
DAILY_RECAP_ENABLED=true
DAILY_RECAP_HOUR=23
DREAM_JOURNAL_ENABLED=true
DREAM_JOURNAL_HOUR=3
MOOD_STATUS_ENABLED=true
MOOD_STATUS_INTERVAL=300
"""
        elif bot_name == "NullVector":
            content += """
RATE_LIMIT_HOURLY=30
RATE_LIMIT_DAILY=100
STM_MESSAGES=8
LTM_SUMMARY_THRESHOLD=6
MAX_MEMORY=50
"""
        env_path.write_text(content)

    ok(f"Created .env at {env_path}")

    if not discord_token:
        warn(f"{bot_name} DISCORD_TOKEN is empty! Add it to .env before starting.")


# ══════════════════════════════════════════════════════════
#  STEP 7: Create Data Directories
# ══════════════════════════════════════════════════════════

def create_data_dir(bot_dir: Path, bot_name: str):
    """Create the data directory for SQLite database."""
    step("Creating data directory", bot_name)

    data_dir = bot_dir / "data"
    data_dir.mkdir(exist_ok=True)
    ok(f"Data directory ready at {data_dir}")

    gitkeep = data_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("")
        ok("Created .gitkeep")


# ══════════════════════════════════════════════════════════
#  STEP 8: Create Systemd User Services
# ══════════════════════════════════════════════════════════

def create_systemd_services(venv_path: Path, bot_dir: Path, bot: BotConfig):
    """Create a systemd user service for a bot."""
    step("Setting up systemd user service", bot.name)

    if not cmd_exists("systemctl"):
        warn("systemctl not found — skipping service setup")
        return

    python_path = venv_path / "bin" / "python"
    bot_script = bot_dir / "bot.py"

    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)

    service_content = f"""[Unit]
Description={bot.name} Discord Bot — {bot.description}
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
SyslogIdentifier={bot.service_name}

# Resource limits
MemoryMax=512M
CPUQuota=80%

[Install]
WantedBy=default.target
"""

    service_file = systemd_dir / f"{bot.service_name}.service"
    service_file.write_text(service_content)
    ok(f"Created service file at {service_file}")

    # Reload systemd
    run(["systemctl", "--user", "daemon-reload"], check=False)
    ok("Reloaded systemd user daemon")


def enable_systemd_services():
    """Enable systemd services and lingering after all services are created."""
    step("Enabling systemd services")

    if not cmd_exists("systemctl"):
        warn("systemctl not found — skipping")
        return

    # Enable lingering
    if cmd_exists("loginctl"):
        run(["loginctl", "enable-linger", os.environ.get("USER", "")], check=False)
        ok("Enabled login linger (bots run even when you're logged out)")

    # Reload daemon
    run(["systemctl", "--user", "daemon-reload"], check=False)

    for bot in BOTS:
        if prompt_yesno(f"Enable {bot.name} to start on boot?", default=True):
            run(["systemctl", "--user", "enable", bot.service_name], check=False)
            ok(f"Enabled {bot.service_name} (auto-start on boot)")
        else:
            info(f"Auto-start not enabled for {bot.name}. Enable later with:")
            info(f"  systemctl --user enable {bot.service_name}")


# ══════════════════════════════════════════════════════════
#  STEP 9: Validate Installations
# ══════════════════════════════════════════════════════════

def validate_installation(venv_path: Path, bot_dir: Path, bot_name: str):
    """Validate that a bot installation looks correct."""
    step("Validating installation", bot_name)

    python_path = venv_path / "bin" / "python"
    checks_passed = 0
    checks_total = 0

    # Check 1: venv exists
    checks_total += 1
    if venv_path.exists():
        ok("Virtual environment exists")
        checks_passed += 1
    else:
        warn("Virtual environment not found")

    # Check 2: .env exists
    checks_total += 1
    if (bot_dir / ".env").exists():
        ok(".env file exists")
        checks_passed += 1
    else:
        warn(".env file not found")

    # Check 3: Discord token is set
    checks_total += 1
    env_path = bot_dir / ".env"
    if env_path.exists():
        content = env_path.read_text()
        for line in content.splitlines():
            if line.startswith("DISCORD_TOKEN="):
                token_val = line.split("=", 1)[1].strip()
                if token_val and token_val != "YOUR_DISCORD_BOT_TOKEN_HERE":
                    ok("Discord token is configured")
                    checks_passed += 1
                else:
                    warn("Discord token is empty in .env")
                break

    # Check 4: Can import discord.py
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

    # Check 5: bot.py exists
    checks_total += 1
    if (bot_dir / "bot.py").exists():
        ok("bot.py exists")
        checks_passed += 1
    else:
        warn("bot.py not found")

    # Check 6: data/ directory
    checks_total += 1
    if (bot_dir / "data").exists():
        ok("Data directory exists")
        checks_passed += 1
    else:
        warn("Data directory not found")

    print(f"\n  {C.BOLD}{bot_name} Validation: {checks_passed}/{checks_total} checks passed{C.RESET}")
    if checks_passed == checks_total:
        ok(f"{bot_name} is ready to go!")
    else:
        warn("Some checks failed. Review the warnings above.")


# ══════════════════════════════════════════════════════════
#  STEP 10: Start Bots
# ══════════════════════════════════════════════════════════

def start_bots(bot_dirs: dict):
    """Start the bots via systemd or offer manual start."""
    step("Starting bots")

    if cmd_exists("systemctl"):
        for bot in BOTS:
            if bot.name not in bot_dirs:
                continue
            service_file = Path.home() / ".config" / "systemd" / "user" / f"{bot.service_name}.service"
            if service_file.exists():
                if prompt_yesno(f"Start {bot.name} via systemd?", default=True):
                    run(["systemctl", "--user", "start", bot.service_name], check=False)
                    ok(f"Started {bot.service_name}")
                    run(["systemctl", "--user", "status", bot.service_name], check=False)

    # Offer manual start
    print(f"\n  {C.BOLD}To start bots manually:{C.RESET}")
    for bot in BOTS:
        if bot.name not in bot_dirs:
            continue
        bot_dir = bot_dirs[bot.name]
        venv_path = bot_dir / VENV_DIR
        python_path = venv_path / "bin" / "python"
        print(f"    {bot.color}{C.BOLD}{bot.name}{C.RESET}: cd {bot_dir} && {python_path} bot.py")

    print(f"\n  {C.BOLD}To manage bots via systemd:{C.RESET}")
    for bot in BOTS:
        print(f"    {bot.color}{C.BOLD}{bot.name}{C.RESET}: systemctl --user start/stop/restart/status {bot.service_name}")

    print(f"\n  {C.BOLD}To view logs:{C.RESET}")
    for bot in BOTS:
        print(f"    {bot.color}{C.BOLD}{bot.name}{C.RESET}: journalctl --user -u {bot.service_name} -f")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    banner()

    # ── Prerequisites ────────────────────────────────────
    check_prerequisites()

    # ── Shared configuration ─────────────────────────────
    shared = collect_shared_config()
    install_dir = shared["install_dir"]
    install_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each bot ─────────────────────────────────
    bot_dirs = {}  # bot_name -> bot_dir Path
    bot_venvs = {}  # bot_name -> venv_path Path

    for bot in BOTS:
        print(f"\n{'='*60}")
        print(f"  {bot.color}{C.BOLD}🌸 Setting up {bot.name} — {bot.description}{C.RESET}")
        print(f"{'='*60}")

        # Clone
        repo_path = clone_repo(bot, install_dir)

        # Determine bot directory
        if bot.bot_subdir == ".":
            bot_dir = repo_path
        else:
            bot_dir = repo_path / bot.bot_subdir

        if not bot_dir.exists():
            fail(f"Bot directory not found at {bot_dir}")

        bot_dirs[bot.name] = bot_dir

        # Create venv
        venv_path = create_venv(bot_dir, bot.name)
        bot_venvs[bot.name] = venv_path

        # Install deps
        install_deps(venv_path, bot_dir, bot.name)

        # Create .env
        create_env(bot_dir, bot.name, shared, bot)

        # Create data dir
        create_data_dir(bot_dir, bot.name)

        # Create systemd service
        create_systemd_services(venv_path, bot_dir, bot)

        # Validate
        validate_installation(venv_path, bot_dir, bot.name)

    # ── Enable systemd services ──────────────────────────
    enable_systemd_services()

    # ── Start bots ───────────────────────────────────────
    start_bots(bot_dirs)

    # ── Final summary ────────────────────────────────────
    print(f"""
{C.PINK}{C.BOLD}
    ╔═══════════════════════════════════════════════╗
    ║   🌸 Lily + NullVector Setup Complete! 🌸     ║
    ╚═══════════════════════════════════════════════╝
{C.RESET}""")

    for bot in BOTS:
        if bot.name not in bot_dirs:
            continue
        bot_dir = bot_dirs[bot.name]
        venv_path = bot_venvs[bot.name]
        print(f"""  {bot.color}{C.BOLD}{bot.name}{C.RESET}
    Directory:  {bot_dir}
    Venv:       {venv_path}
    Config:     {bot_dir / '.env'}
    Database:   {bot_dir / 'data' / ('lily.db' if bot.name == 'Lily' else 'nullvector.db')}
    Service:    {bot.service_name}
""")

    print(f"""  {C.BOLD}Quick commands:{C.RESET}
    Lily start:      systemctl --user start lily-bot
    NullVector start: systemctl --user start nullvector-bot
    Lily logs:       journalctl --user -u lily-bot -f
    NullVector logs: journalctl --user -u nullvector-bot -f
    Stop all:        systemctl --user stop lily-bot nullvector-bot
    Restart all:     systemctl --user restart lily-bot nullvector-bot

  {C.DIM}Lily v8.5 — She lives 💕 | NullVector v2.0 — Smart routing 🧠{C.RESET}
""")


if __name__ == "__main__":
    main()
