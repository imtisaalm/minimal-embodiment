// Copy this file to `secrets.h` in the same folder and fill in real values.
// `secrets.h` is gitignored — it never gets pushed to GitHub.
#pragma once

#define SECRET_WIFI_SSID     "YOUR_WIFI_SSID"
#define SECRET_WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// Must match the US_BRIDGE_TOKEN the bridge is started with.
#define SECRET_BRIDGE_TOKEN  "YOUR_BRIDGE_TOKEN"

// Local mode (BRIDGE_USE_TLS = false in the .ino): your laptop's LAN address
// and the bridge port, e.g. "192.168.1.42:3737".
// Tunnel mode (BRIDGE_USE_TLS = true): public hostname only, e.g.
// "bridge.yourdomain.com".
#define SECRET_BRIDGE_HOST   "192.168.1.42:3737"
