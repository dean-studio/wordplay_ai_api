module.exports = {
  apps: [
    {
      name: "kyobo-api",
      script: "/home/dean/fastapi/venv/bin/python",
      args: "-m uvicorn api:app --host 0.0.0.0 --port 8000",
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
