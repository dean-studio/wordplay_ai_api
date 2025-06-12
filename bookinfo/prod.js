module.exports = {
  apps: [
    {
      name: "kyobo-api",
      script: "python",
      args: ["api.py"],
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G"
    }
  ]
};
