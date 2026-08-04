"""User management CLI. No default admin account is created automatically —
run this to create the first one.

Usage:
  python -m webapp.manage create-user <username> --role admin
  python -m webapp.manage list-users
  python -m webapp.manage set-role <username> <role>
  python -m webapp.manage deactivate <username>
  python -m webapp.manage activate <username>
"""

from __future__ import annotations

import argparse
import getpass
import sys

from .auth import hash_password
from .db import SessionLocal, init_db
from .models import ROLES, User


def cmd_create_user(args: argparse.Namespace) -> int:
    if args.role not in ROLES:
        print(f"error: role must be one of {ROLES}", file=sys.stderr)
        return 1

    if args.password is not None:
        # Non-interactive path for scripts/CI/container entrypoints. Prefer
        # the interactive prompt for anything typed by a human: this flag
        # leaves the password sitting in shell history and process listings.
        password = args.password
    else:
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("error: passwords don't match", file=sys.stderr)
            return 1
    if len(password) < 8:
        print("error: password must be at least 8 characters", file=sys.stderr)
        return 1

    db = SessionLocal()
    try:
        if db.query(User).filter_by(username=args.username).first():
            print(f"error: user {args.username!r} already exists", file=sys.stderr)
            return 1
        user = User(
            username=args.username,
            password_hash=hash_password(password),
            role=args.role,
            is_active=True,
        )
        db.add(user)
        db.commit()
        print(f"Created user {args.username!r} with role {args.role!r}.")
        return 0
    finally:
        db.close()


def cmd_list_users(args: argparse.Namespace) -> int:
    db = SessionLocal()
    try:
        users = db.query(User).order_by(User.id).all()
        if not users:
            print("No users yet. Create one with: python -m webapp.manage create-user <username> --role admin")
            return 0
        for u in users:
            status = "active" if u.is_active else "deactivated"
            print(f"{u.id:>4}  {u.username:<20} {u.role:<10} {status}")
        return 0
    finally:
        db.close()


def cmd_set_role(args: argparse.Namespace) -> int:
    if args.role not in ROLES:
        print(f"error: role must be one of {ROLES}", file=sys.stderr)
        return 1
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(username=args.username).first()
        if not user:
            print(f"error: no such user {args.username!r}", file=sys.stderr)
            return 1
        user.role = args.role
        db.commit()
        print(f"{args.username} is now {args.role}.")
        return 0
    finally:
        db.close()


def cmd_set_active(args: argparse.Namespace, active: bool) -> int:
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(username=args.username).first()
        if not user:
            print(f"error: no such user {args.username!r}", file=sys.stderr)
            return 1
        user.is_active = active
        db.commit()
        print(f"{args.username} {'activated' if active else 'deactivated'}.")
        return 0
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage thermal-panel-inspector user accounts")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create-user", help="Create a new user (prompts for password)")
    p_create.add_argument("username")
    p_create.add_argument("--role", default="inspector", choices=ROLES)
    p_create.add_argument(
        "--password",
        default=None,
        help="Set the password non-interactively (for scripts/CI). Omit to be prompted "
        "securely instead, which is preferred outside of automation.",
    )
    p_create.set_defaults(func=cmd_create_user)

    p_list = sub.add_parser("list-users", help="List all users")
    p_list.set_defaults(func=cmd_list_users)

    p_role = sub.add_parser("set-role", help="Change a user's role")
    p_role.add_argument("username")
    p_role.add_argument("role", choices=ROLES)
    p_role.set_defaults(func=cmd_set_role)

    p_deact = sub.add_parser("deactivate", help="Deactivate a user (blocks login)")
    p_deact.add_argument("username")
    p_deact.set_defaults(func=lambda a: cmd_set_active(a, False))

    p_act = sub.add_parser("activate", help="Reactivate a user")
    p_act.add_argument("username")
    p_act.set_defaults(func=lambda a: cmd_set_active(a, True))

    args = parser.parse_args(argv)
    init_db()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
