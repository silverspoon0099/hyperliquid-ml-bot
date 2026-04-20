// PM2 process file for the ML trading bot supervisor.
//
// Usage:
//   pm2 start ecosystem.config.js                  # start the bot
//   pm2 logs ml-bot                                # tail logs
//   pm2 restart ml-bot                             # restart
//   pm2 stop ml-bot                                # stop (sends SIGINT → graceful shutdown)
//   pm2 save && pm2 startup                       # persist across reboots (one-time)
//
// pm2's default kill signal is SIGINT (configurable below) — bot.py installs
// SIGINT/SIGTERM handlers that drain trade buffers and join service threads
// before exiting. `kill_timeout` gives that drain enough time before pm2
// escalates to SIGKILL.

module.exports = {
  apps: [
    {
      name: "ml-bot",
      cwd: "/nvme1/projects/trading/ml-bot",
      script: "bot.py",
      interpreter: "/nvme1/projects/trading/ml-bot/.venv/bin/python",
      // Pass `--only follower` etc. via args if you ever want partial start.
      args: "",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,           // 5s between restart attempts
      min_uptime: "30s",             // crash-loop guard
      kill_timeout: 20000,           // > supervisor.shutdown_timeout_sec (15s) so drain finishes
      kill_signal: "SIGINT",         // bot.py handles SIGINT for graceful shutdown
      // Loguru already rotates app logs into ./logs/ — pm2 logs are just stdout/stderr.
      out_file: "/nvme1/projects/trading/ml-bot/logs/pm2.out.log",
      error_file: "/nvme1/projects/trading/ml-bot/logs/pm2.err.log",
      merge_logs: true,
      time: true,                    // prepend timestamps to pm2 log lines
      env: {
        PYTHONUNBUFFERED: "1",       // flush prints immediately to pm2 logs
      },
    },
  ],
};
