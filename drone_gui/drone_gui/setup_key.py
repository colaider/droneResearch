"""First-run setup: get this machine an SSH key the drone accepts.

`run.sh --login` / `run.cmd --login` calls this. It makes a key if there is none, installs the
public half on the board, and checks that what the panel needs actually works.

The password is typed into ssh, never into this script: the key is passed as an argument and
ssh reads the password straight from the terminal, so it is never a Python string, never in
argv, and never in a log. The script cannot see it even if it wanted to.

ssh-copy-id would do the same job but is not on Windows, so the equivalent is done by hand.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys

KEY_TYPE = "ed25519"
# Keys the panel will pick up on its own, best first. rb3_claude stays in the list so the
# machines that already have it keep working.
CANDIDATES = ("rb3_drone", "rb3_claude", "id_ed25519", "id_rsa")


def ssh_dir():
    return os.path.join(os.path.expanduser("~"), ".ssh")


def default_key_path():
    """An existing candidate key, else where --login would create one."""
    for name in CANDIDATES:
        path = os.path.join(ssh_dir(), name)
        if os.path.isfile(path):
            return path
    return os.path.join(ssh_dir(), CANDIDATES[0])


def have(program):
    from shutil import which
    return which(program) is not None


def generate(path):
    """Create a keypair at `path`. No passphrase: the panel runs ssh unattended."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.name != "nt":
        try:
            os.chmod(os.path.dirname(path), 0o700)
        except OSError:
            pass
    comment = "drone_gui@%s" % (os.environ.get("COMPUTERNAME") or os.uname().nodename
                                if hasattr(os, "uname") else "windows")
    print("creating %s" % path)
    done = subprocess.run(["ssh-keygen", "-t", KEY_TYPE, "-f", path, "-N", "", "-C", comment])
    if done.returncode != 0 or not os.path.isfile(path):
        return False
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return True


def key_works(target, path, timeout=10):
    """Does key-only login already work? BatchMode so it fails instead of prompting."""
    done = subprocess.run(
        ["ssh", "-i", path, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         "-o", "ConnectTimeout=%d" % timeout, target, "true"],
        capture_output=True, text=True)
    return done.returncode == 0


def append_command(public_key):
    """The shell one-liner that adds a key to authorized_keys, idempotently.

    grep -qxF first so running --login twice does not duplicate the line, and the key is
    quoted as one argument so nothing in the comment field can be interpreted by the shell.
    """
    quoted = shlex.quote(public_key)
    return (
        "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; "
        "grep -qxF %s ~/.ssh/authorized_keys || printf '%%s\\n' %s >> ~/.ssh/authorized_keys; "
        "chmod 600 ~/.ssh/authorized_keys; echo INSTALLED"
    ) % (quoted, quoted)


def install(target, path):
    """Append the public key to the board's authorized_keys, over a password login.

    The key goes in as a quoted argument rather than on stdin, so ssh keeps the terminal for
    its own password prompt and nothing has to be typed twice.
    """
    public_path = path + ".pub"
    try:
        with open(public_path, encoding="utf-8") as handle:
            public_key = handle.read().strip()
    except OSError as exc:
        print("cannot read %s: %s" % (public_path, exc))
        return False
    if not public_key:
        print("%s is empty" % public_path)
        return False

    quoted = shlex.quote(public_key)
    remote = (
        "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; "
        "grep -qxF %s ~/.ssh/authorized_keys || printf '%%s\\n' %s >> ~/.ssh/authorized_keys; "
        "chmod 600 ~/.ssh/authorized_keys; echo INSTALLED"
    ) % (quoted, quoted)

    print("\nInstalling this machine's public key on %s." % target)
    print("The board will ask for the password of the '%s' account - ssh prompts for it "
          "directly,\nit is not read or stored by this script.\n" % target.split("@")[0])
    # Inherit the terminal: ssh needs it for the prompt.
    done = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=accept-new",
         # Ignore keys that are already loaded, or ssh may keep retrying a key the board
         # rejects and never get round to asking for the password.
         "-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password,keyboard-interactive",
         target, remote])
    return done.returncode == 0


def check_sudo(target, path):
    """The panel's start/stop buttons need passwordless sudo for systemctl."""
    done = subprocess.run(
        ["ssh", "-i", path, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target,
         "sudo -n true"], capture_output=True, text=True)
    return done.returncode == 0


def ensure_access(config, force=False):
    """Make `config.ssh_key` a key the board accepts. True if the panel can go ahead."""
    target, path = config.ssh_target, config.ssh_key

    for program in ("ssh", "ssh-keygen"):
        if not have(program):
            print("%s is not installed or not on PATH.\n"
                  "  Windows: Settings > System > Optional features > OpenSSH Client\n"
                  "  Debian/Ubuntu: sudo apt install openssh-client" % program)
            return False

    if not os.path.isfile(path):
        print("no key at %s" % path)
        if not generate(path):
            print("could not create a key")
            return False
    else:
        print("using key %s" % path)

    if not force and key_works(target, path):
        print("key already works on %s" % target)
    else:
        if not install(target, path):
            print("\nkey install failed - was the password right, and does %s allow password "
                  "logins?" % config.host)
            return False
        if not key_works(target, path):
            print("\nthe key was installed but login still fails. Check the board's "
                  "/var/log/auth.log; a home directory or ~/.ssh with group write permission "
                  "makes sshd refuse the key.")
            return False
        print("key installed - %s now logs in without a password" % target)

    if check_sudo(target, path):
        print("passwordless sudo: yes (start/stop/restart will work)")
    else:
        print("passwordless sudo: NO - telemetry and the camera will work, but the panel "
              "cannot start or stop services.\n"
              "  Fix on the board:  echo \"$USER ALL=(ALL) NOPASSWD: ALL\" | "
              "sudo tee /etc/sudoers.d/90-$USER")
    return True


def main(argv=None):
    """Standalone: python3 -m drone_gui.setup_key [--host H] [--ssh-key PATH] [--force]"""
    import argparse

    from .config import CONFIG

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=CONFIG.host)
    parser.add_argument("--user", default=CONFIG.user)
    parser.add_argument("--ssh-key", default=CONFIG.ssh_key)
    parser.add_argument("--force", action="store_true",
                        help="install the key even if login already works")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    CONFIG.host, CONFIG.user, CONFIG.ssh_key = args.host, args.user, args.ssh_key
    return 0 if ensure_access(CONFIG, force=args.force) else 1


if __name__ == "__main__":
    raise SystemExit(main())
