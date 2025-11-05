from datetime import datetime


def create_health_response(status):
    """Create health response."""
    return {
        "status": status,
    }


def create_error_response(error, details=None):
    """Create error response."""
    return {
        "error": error,
        "details": details,
        "timestamp": datetime.utcnow().isoformat(),
    }
