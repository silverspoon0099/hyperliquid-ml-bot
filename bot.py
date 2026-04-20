"""ML trading bot supervisor — modeled on dataverse/main/neurons/miner.py.

Reads `services:` from config.yaml and starts each enabled service as a
daemon thread. Postgres is the coordination substrate: services don't talk
to each other directly, they read/write the same hypertables.

Each service module exposes a `run(stop_event)` entry point. The supervisor
wires a shared `threading.Event`; on SIGINT/SIGTERM it sets the event and
joins each thread (with `supervisor.shutdown_timeout_sec` budget per thread).

Add a new service:
  1. Implement `def run(stop_event: threading.Event) -> None` in your module.
  2. Append an entry to `_SERVICES` below mapping config key → import path.
  3. Add `<key>: { enabled: true, ... }` to config.yaml under `services:`.

Run:
    python bot.py                # all enabled services
    python bot.py --only follower hl_collector
    python bot.py --dry-run      # print service plan, don't start anything
"""
from __future__ import annotations

import argparse
import importlib
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

from data import db  # noqa: E402  — dotenv must load first
from utils.config import load_config  # noqa: E402
from utils.logging_setup import get_logger  # noqa: E402

log = get_logger("bot")


# ─────────────────────────────────────────────────────────────────────────
# Service registry — config key → "module.path:callable"
# Add a new service by appending a line. The callable must accept a
# threading.Event named `stop_event` (positional or keyword).
# ─────────────────────────────────────────────────────────────────────────
_SERVICES: dict[str, str] = {
    "follower":     "scripts.follower:run",
    "hl_collector": "data.collectors.hyperliquid_ws:run",
    # Phase 2 placeholders — wire up when modules exist:
    # "predictor":  "model.predictor:run",
    # "executor":   "execution.executor:run",
}


@dataclass
class ServicePlan:
    name: str
    target: Callable
    kwargs: dict


def _resolve(spec: str) -> Callable:
    """Resolve "pkg.mod:func" → callable. Raises ImportError if missing."""
    module_path, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"service spec '{spec}' missing ':callable' suffix")
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def _build_plan(cfg: dict, only: list[str] | None) -> list[ServicePlan]:
    services_cfg = cfg.get("services", {}) or {}
    plan: list[ServicePlan] = []
    for name, spec in _SERVICES.items():
        if only and name not in only:
            continue
        scfg = services_cfg.get(name, {}) or {}
        if not scfg.get("enabled", False):
            log.info(f"service '{name}': disabled, skipping")
            continue
        try:
            target = _resolve(spec)
        except (ImportError, AttributeError, ValueError) as exc:
            log.error(f"service '{name}': cannot load '{spec}': {exc}")
            continue
        plan.append(ServicePlan(name=name, target=target, kwargs={}))
    return plan


class Bot:
    """The Glorious ML Trading Bot supervisor."""

    def __init__(self, plan: list[ServicePlan], shutdown_timeout_sec: int, health_log_interval_sec: int):
        self.plan = plan
        self.shutdown_timeout_sec = shutdown_timeout_sec
        self.health_log_interval_sec = health_log_interval_sec
        self.stop_event = threading.Event()
        self.threads: dict[str, threading.Thread] = {}
        self._shutdown_lock = threading.Lock()
        self._shutting_down = False

    def _service_wrapper(self, name: str, target: Callable, kwargs: dict) -> None:
        """Thin wrapper that logs lifecycle + crashes. Sets stop_event on
        unhandled crash so the supervisor exits — Postgres remains the source
        of truth so a partial restart is the safe default (re-launch under
        systemd / process manager)."""
        log.info(f"[{name}] starting")
        try:
            target(stop_event=self.stop_event, **kwargs)
        except Exception as exc:
            log.exception(f"[{name}] CRASHED: {exc} — signalling shutdown")
            self.stop_event.set()
        else:
            log.info(f"[{name}] exited cleanly")

    def start(self) -> None:
        # Verify DB before spawning anything — fail fast on misconfig.
        info = db.ping()
        log.info(
            f"DB ok: {info['user']}@{info['database']} (timescaledb {info['timescaledb']})"
        )
        log.info(f"Starting {len(self.plan)} service(s): {[p.name for p in self.plan]}")

        for p in self.plan:
            t = threading.Thread(
                target=self._service_wrapper,
                args=(p.name, p.target, p.kwargs),
                name=p.name,
                daemon=True,
            )
            t.start()
            self.threads[p.name] = t

    def wait(self) -> None:
        """Block until stop_event is set, emitting a heartbeat periodically."""
        last_health = time.time()
        while not self.stop_event.is_set():
            time.sleep(0.5)
            now = time.time()
            if now - last_health >= self.health_log_interval_sec:
                alive = [n for n, t in self.threads.items() if t.is_alive()]
                dead = [n for n, t in self.threads.items() if not t.is_alive()]
                log.info(f"heartbeat: alive={alive} dead={dead}")
                if dead and not self.stop_event.is_set():
                    log.error(f"service(s) {dead} died unexpectedly — initiating shutdown")
                    self.stop_event.set()
                last_health = now

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True
        log.info("Shutdown requested — signalling services and joining threads…")
        self.stop_event.set()
        for name, t in self.threads.items():
            t.join(timeout=self.shutdown_timeout_sec)
            if t.is_alive():
                log.warning(f"[{name}] did not exit within {self.shutdown_timeout_sec}s — abandoning (daemon)")
            else:
                log.info(f"[{name}] joined")
        log.info("Bot stopped.")


def _install_signal_handlers(bot: Bot) -> None:
    def handler(signum, _frame):
        log.info(f"Received signal {signum}")
        bot.stop_event.set()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="ML trading bot supervisor")
    parser.add_argument("--only", nargs="+", help="run only the named services (space-separated)")
    parser.add_argument("--dry-run", action="store_true", help="print plan and exit")
    args = parser.parse_args()

    cfg = load_config()
    sup_cfg = cfg.get("supervisor", {}) or {}

    plan = _build_plan(cfg, only=args.only)

    if args.dry_run:
        log.info("DRY RUN — would start:")
        for p in plan:
            log.info(f"  {p.name}: {p.target.__module__}.{p.target.__name__}")
        sys.exit(0)

    if not plan:
        log.error("No services to start — check `services:` in config.yaml")
        sys.exit(2)

    bot = Bot(
        plan=plan,
        shutdown_timeout_sec=int(sup_cfg.get("shutdown_timeout_sec", 15)),
        health_log_interval_sec=int(sup_cfg.get("health_log_interval_sec", 60)),
    )
    _install_signal_handlers(bot)

    try:
        bot.start()
        bot.wait()
    finally:
        bot.shutdown()


if __name__ == "__main__":
    main()
