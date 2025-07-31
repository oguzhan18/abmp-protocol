"""
ABMP Protocol Custom Exceptions
Clean error handling with specific exception types
"""

from typing import Optional, Any


class ABMPError(Exception):
    """Base exception for all ABMP protocol errors"""
    
    def __init__(self, message: str, error_code: str = None, context: dict = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class NetworkError(ABMPError):
    """Network-related errors"""
    pass


class RoutingError(NetworkError):
    """Routing-specific errors"""
    pass


class NodeNotFoundError(RoutingError):
    """Node not found in network"""
    
    def __init__(self, node_id: int):
        super().__init__(
            f"Node {node_id} not found in network",
            "NODE_NOT_FOUND",
            {"node_id": node_id}
        )
        self.node_id = node_id


class RouteNotFoundError(RoutingError):
    """No route found to destination"""
    
    def __init__(self, source: int, destination: int):
        super().__init__(
            f"No route found from {source} to {destination}",
            "ROUTE_NOT_FOUND",
            {"source": source, "destination": destination}
        )
        self.source = source
        self.destination = destination


class PacketError(NetworkError):
    """Packet processing errors"""
    pass


class InvalidPacketError(PacketError):
    """Invalid packet format or content"""
    
    def __init__(self, reason: str, packet_data: Any = None):
        super().__init__(
            f"Invalid packet: {reason}",
            "INVALID_PACKET",
            {"reason": reason, "packet_data": str(packet_data)}
        )


class PacketTooLargeError(PacketError):
    """Packet exceeds maximum size"""
    
    def __init__(self, size: int, max_size: int):
        super().__init__(
            f"Packet size {size} exceeds maximum {max_size}",
            "PACKET_TOO_LARGE",
            {"size": size, "max_size": max_size}
        )


class SecurityError(ABMPError):
    """Security-related errors"""
    pass


class AuthenticationError(SecurityError):
    """Authentication failure"""
    pass


class EncryptionError(SecurityError):
    """Encryption/decryption errors"""
    pass


class InvalidSignatureError(SecurityError):
    """Digital signature verification failed"""
    
    def __init__(self, signer_id: int = None):
        super().__init__(
            f"Invalid signature from node {signer_id}",
            "INVALID_SIGNATURE",
            {"signer_id": signer_id}
        )


class HealingError(ABMPError):
    """Self-healing system errors"""
    pass


class AnomalyDetectionError(HealingError):
    """Anomaly detection system errors"""
    pass


class HealingActionError(HealingError):
    """Healing action execution errors"""
    
    def __init__(self, action: str, reason: str):
        super().__init__(
            f"Healing action '{action}' failed: {reason}",
            "HEALING_ACTION_FAILED",
            {"action": action, "reason": reason}
        )


class ConfigurationError(ABMPError):
    """Configuration-related errors"""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration parameters"""
    
    def __init__(self, parameter: str, value: Any, expected: str):
        super().__init__(
            f"Invalid configuration: {parameter}={value}, expected {expected}",
            "INVALID_CONFIGURATION",
            {"parameter": parameter, "value": value, "expected": expected}
        )


class ResourceError(ABMPError):
    """Resource-related errors"""
    pass


class InsufficientResourcesError(ResourceError):
    """Insufficient system resources"""
    
    def __init__(self, resource_type: str, required: Any, available: Any):
        super().__init__(
            f"Insufficient {resource_type}: required {required}, available {available}",
            "INSUFFICIENT_RESOURCES",
            {"resource_type": resource_type, "required": required, "available": available}
        )


class TimeoutError(ABMPError):
    """Operation timeout errors"""
    
    def __init__(self, operation: str, timeout: float):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout} seconds",
            "OPERATION_TIMEOUT",
            {"operation": operation, "timeout": timeout}
        )


class StateError(ABMPError):
    """Invalid state transition errors"""
    
    def __init__(self, current_state: str, attempted_transition: str):
        super().__init__(
            f"Invalid state transition from '{current_state}' to '{attempted_transition}'",
            "INVALID_STATE_TRANSITION",
            {"current_state": current_state, "attempted_transition": attempted_transition}
        )


class CryptoError(SecurityError):
    """Cryptographic operation errors"""
    pass


class KeyGenerationError(CryptoError):
    """Key generation errors"""
    pass


class KeyNotFoundError(CryptoError):
    """Cryptographic key not found"""
    
    def __init__(self, key_id: str):
        super().__init__(
            f"Cryptographic key '{key_id}' not found",
            "KEY_NOT_FOUND",
            {"key_id": key_id}
        )


class QuantumError(CryptoError):
    """Quantum cryptography errors"""
    pass


class QKDError(QuantumError):
    """Quantum Key Distribution errors"""
    
    def __init__(self, error_rate: float, threshold: float):
        super().__init__(
            f"QKD error rate {error_rate:.3f} exceeds threshold {threshold:.3f}",
            "QKD_ERROR_RATE_HIGH",
            {"error_rate": error_rate, "threshold": threshold}
        )


# Exception handling utilities
def handle_abmp_exception(func):
    """Decorator for ABMP exception handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ABMPError:
            raise  # Re-raise ABMP exceptions as-is
        except Exception as e:
            # Wrap unknown exceptions
            raise ABMPError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                "UNEXPECTED_ERROR",
                {"function": func.__name__, "original_error": str(e)}
            ) from e
    return wrapper