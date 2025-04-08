# middleware/csp_headers.py

class CSPFrameAllowMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response['Content-Security-Policy'] = "frame-ancestors http://localhost:3000"
        return response
