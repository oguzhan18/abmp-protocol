"""
ABMP Protocol Core Interfaces
Global Production Protocol - Enterprise Architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import time


class Priority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 128
    HIGH = 192
    CRITICAL = 255


class NodeState(Enum):
    """Node operational states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class NodeMetrics:
    """Immutable node performance metrics"""
    node_id: int
    cpu_usage: float
    memory_usage: float
    network_latency: float
    packet_loss_rate: float
    uptime: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', time.time())


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable routing decision result"""
    next_hop: Optional[int]
    confidence: float
    alternative_hops: List[int]
    reasoning: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', time.time())


@runtime_checkable
class INetworkNode(Protocol):
    """Network node interface"""
    
    @property
    def node_id(self) -> int:
        ...
    
    @property
    def state(self) -> NodeState:
        ...
    
    def send_message(self, destination: int, data: bytes, priority: Priority = Priority.NORMAL) -> bool:
        ...
    
    def get_metrics(self) -> NodeMetrics:
        ...


@runtime_checkable
class IRoutingEngine(Protocol):
    """Routing engine interface"""
    
    def find_route(self, destination: int, current_metrics: NodeMetrics) -> RoutingDecision:
        ...
    
    def update_routing_table(self, node_id: int, metrics: NodeMetrics) -> None:
        ...
    
    def handle_route_failure(self, failed_route: int) -> None:
        ...


@runtime_checkable
class IHealingEngine(Protocol):
    """Self-healing engine interface"""
    
    def detect_anomaly(self, metrics: NodeMetrics) -> bool:
        ...
    
    def suggest_healing_action(self, anomaly_type: str) -> List[str]:
        ...
    
    def execute_healing(self, action: str) -> bool:
        ...


@runtime_checkable
class ICryptoEngine(Protocol):
    """Cryptographic engine interface"""
    
    def encrypt(self, data: bytes, recipient_id: int) -> bytes:
        ...
    
    def decrypt(self, encrypted_data: bytes, sender_id: int) -> bytes:
        ...
    
    def sign(self, data: bytes) -> bytes:
        ...
    
    def verify(self, data: bytes, signature: bytes, signer_id: int) -> bool:
        ...


class IObserver(ABC):
    """Observer pattern interface"""
    
    @abstractmethod
    def update(self, event_type: str, event_data: Dict[str, Any]) -> None:
        pass


class IEventPublisher(ABC):
    """Publisher interface"""
    
    @abstractmethod
    def subscribe(self, observer: IObserver, event_type: str) -> None:
        pass
    
    @abstractmethod
    def unsubscribe(self, observer: IObserver, event_type: str) -> None:
        pass
    
    @abstractmethod
    def notify(self, event_type: str, event_data: Dict[str, Any]) -> None:
        pass


class IPacketHandler(ABC):
    """Abstract packet handler"""
    
    def __init__(self):
        self._next_handler: Optional['IPacketHandler'] = None
    
    def set_next(self, handler: 'IPacketHandler') -> 'IPacketHandler':
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, packet: Any) -> bool:
        pass
    
    def _handle_next(self, packet: Any) -> bool:
        if self._next_handler:
            return self._next_handler.handle(packet)
        return False


class IRoutingStrategy(ABC):
    """Strategy pattern interface"""
    
    @abstractmethod
    def calculate_route(self, source: int, destination: int, 
                       network_state: Dict[str, Any]) -> RoutingDecision:
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        pass


class INodeFactory(ABC):
    """Factory pattern interface"""
    
    @abstractmethod
    def create_node(self, node_type: str, node_id: int, **kwargs) -> INetworkNode:
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        pass


class ICommand(ABC):
    """Command pattern interface"""
    
    @abstractmethod
    def execute(self) -> bool:
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        pass
    
    @abstractmethod
    def can_undo(self) -> bool:
        pass


class INetworkTopology(ABC):
    """Network topology abstraction"""
    
    @abstractmethod
    def add_node(self, node_id: int) -> None:
        pass
    
    @abstractmethod
    def remove_node(self, node_id: int) -> None:
        pass
    
    @abstractmethod
    def add_link(self, node1: int, node2: int, weight: float = 1.0) -> None:
        pass
    
    @abstractmethod
    def remove_link(self, node1: int, node2: int) -> None:
        pass
    
    @abstractmethod
    def get_neighbors(self, node_id: int) -> List[int]:
        pass
    
    @abstractmethod
    def get_shortest_path(self, source: int, destination: int) -> List[int]:
        pass


NodeID = int
PacketID = str
RoutingTable = Dict[NodeID, List[NodeID]]
NetworkState = Dict[str, Any]
EventData = Dict[str, Any]