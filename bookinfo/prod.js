module.exports = {
  apps: [
    {
      name: "preview-updater",
      script: "/home/dean/fastapi/venv/bin/python",
      args: "api.py",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "production",
        PYTHONPATH: "."
      },
      error_file: "./logs/err.log",
      out_file: "./logs/out.log",
      log_file: "./logs/combined.log",
      time: true
    }
  ]
};
