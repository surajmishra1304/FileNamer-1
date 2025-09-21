app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],   # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],   # Allow all headers
)