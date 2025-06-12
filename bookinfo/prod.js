module.exports = {
  apps: [
    {
      name: "kyobo-api",
      script: "python",
      args: "api.py",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "production",
        PATH: "/home/dean/fastapi/venv/bin:" + process.env.PATH
      },
      error_file: "./logs/err.log",
      out_file: "./logs/out.log",
      log_file: "./logs/combined.log",
      time: true
    }
  ]
};
