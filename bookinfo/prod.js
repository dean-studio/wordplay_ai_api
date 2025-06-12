module.exports = {
  apps: [
    {
      name: "kyobo-api",
      script: "/home/dean/fastapi/venv/bin/python",
      args: "api.py",
      instances: 1,
      autorestart: true,
      watch: false,
      error_file: "./logs/err.log",
      out_file: "./logs/out.log",
      log_file: "./logs/combined.log",
      time: true
    }
  ]
};
