import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

_data_json = ""
_HTML = (Path(__file__).with_name("profiler_ui.html")).read_text()

class _Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == "/api/data":
      self._json(200, _data_json)
    elif self.path == "/" or self.path == "/index.html":
      self._html(200, _HTML)
    else:
      self._json(404, '{"error":"not found"}')

  def _json(self, code, body):
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.end_headers()
    self.wfile.write(body.encode() if isinstance(body, str) else body)

  def _html(self, code, body):
    self.send_response(code)
    self.send_header("Content-Type", "text/html")
    self.end_headers()
    self.wfile.write(body.encode())

  def log_message(self, *_):
    pass

def serve(data: dict, port: int = 8000):
  global _data_json
  _data_json = json.dumps(data)
  server = HTTPServer(("", port), _Handler)
  print(f"open profiler @ http://localhost:{port}")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()
