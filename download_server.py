#!/usr/bin/env python3
import http.server
import socketserver
import os

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

os.chdir('/workspace')
print(f"服务器启动在端口 {PORT}")
print(f"文件列表:")
for file in os.listdir('.'):
    if file.endswith('.tar.gz'):
        print(f"  - http://localhost:{PORT}/{file}")

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"\n在浏览器中访问: http://localhost:{PORT}/")
    print("按 Ctrl+C 停止服务器")
    httpd.serve_forever()