"""
ABMP Event System
Observer Pattern Implementation for Network Events
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import logging
from collections import defaultdict

from interfaces import IObserver, IEventPublisher
from exceptions import ABMPError


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Network event types"""
    NODE_JOINED = "node_joined"
    NODE_LEFT = "node_left"
    NODE_FAILED = "node_failed"
    ROUTE_DISCOVERED = "route_discovered"
    ROUTE_FAILED = "route_failed"
    PACKET_SENT = "packet_sent"
    PACKET_RECEIVED = "packet_received"
    HEALING_TRIGGERED = "healing_triggered"
    HEALING_COMPLETED = "healing_completed"
    ANOMALY_DETECTED = "anomaly_detected"
    SECURITY_ALERT = "security_alert"
    PERFORMANCE_ALERT = "performance_alert"
    TOPOLOGY_CHANGED = "topology_changed"


@dataclass(frozen=True)
class NetworkEvent:
    """Immutable network event"""
    event_type: EventType
    source_node: int
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"evt_{int(time.time() * 1000000)}")
    
    def __post_init__(self):
        if not self.timestamp:
            object.__setattr__(self, 'timestamp', time.time())


class EventBus(IEventPublisher):
    """
    Thread-safe event bus implementation
    Follows Observer pattern and Dependency Inversion Principle
    """
    
    def __init__(self):
        self._observers: Dict[EventType, List[IObserver]] = defaultdict(list)
        self._wildcard_observers: List[IObserver] = []
        self._event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: List[NetworkEvent] = []
        self._max_history_size = 1000
        self._lock = threading.RLock()
        
    def subscribe(self, observer: IObserver, event_type: str) -> None:
        """Subscribe observer to specific event type"""
        with self._lock:
            try:
                event_enum = EventType(event_type)
                if observer not in self._observers[event_enum]:
                    self._observers[event_enum].append(observer)
                    logger.debug(f"Observer {observer} subscribed to {event_type}")
            except ValueError:
                raise ABMPError(f"Invalid event type: {event_type}")
    
    def subscribe_all(self, observer: IObserver) -> None:
        """Subscribe observer to all event types"""
        with self._lock:
            if observer not in self._wildcard_observers:
                self._wildcard_observers.append(observer)
                logger.debug(f"Observer {observer} subscribed to all events")
    
    def unsubscribe(self, observer: IObserver, event_type: str) -> None:
        """Unsubscribe observer from specific event type"""
        with self._lock:
            try:
                event_enum = EventType(event_type)
                if observer in self._observers[event_enum]:
                    self._observers[event_enum].remove(observer)
                    logger.debug(f"Observer {observer} unsubscribed from {event_type}")
            except ValueError:
                raise ABMPError(f"Invalid event type: {event_type}")
    
    def unsubscribe_all(self, observer: IObserver) -> None:
        """Unsubscribe observer from all event types"""
        with self._lock:
            # Remove from specific event subscriptions
            for observers_list in self._observers.values():
                if observer in observers_list:
                    observers_list.remove(observer)
            
            # Remove from wildcard subscriptions
            if observer in self._wildcard_observers:
                self._wildcard_observers.remove(observer)
            
            logger.debug(f"Observer {observer} unsubscribed from all events")
    
    def add_handler(self, event_type: EventType, handler: Callable[[NetworkEvent], None]) -> None:
        """Add function-based event handler"""
        with self._lock:
            self._event_handlers[event_type].append(handler)
            logger.debug(f"Handler {handler.__name__} added for {event_type.value}")
    
    def notify(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Notify all subscribers of event"""
        try:
            event_enum = EventType(event_type)
            event = NetworkEvent(
                event_type=event_enum,
                source_node=event_data.get('source_node', -1),
                timestamp=time.time(),
                data=event_data
            )
            self.publish_event(event)
        except ValueError:
            raise ABMPError(f"Invalid event type: {event_type}")
    
    def publish_event(self, event: NetworkEvent) -> None:
        """Publish event to all subscribers"""
        with self._lock:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history_size:
                self._event_history = self._event_history[-self._max_history_size:]
            
            # Notify observers
            self._notify_observers(event)
            
            # Call handlers
            self._call_handlers(event)
    
    def _notify_observers(self, event: NetworkEvent) -> None:
        """Notify all observers of event"""
        observers_to_notify = []
        
        # Add specific event observers
        observers_to_notify.extend(self._observers[event.event_type])
        
        # Add wildcard observers
        observers_to_notify.extend(self._wildcard_observers)
        
        # Notify all observers (outside lock to avoid deadlock)
        failed_observers = []
        for observer in observers_to_notify:
            try:
                observer.update(event.event_type.value, event.data)
            except Exception as e:
                logger.error(f"Observer {observer} failed to handle event {event.event_type}: {e}")
                failed_observers.append(observer)
        
        # Remove failed observers
        if failed_observers:
            with self._lock:
                for observer in failed_observers:
                    self.unsubscribe_all(observer)
    
    def _call_handlers(self, event: NetworkEvent) -> None:
        """Call all handlers for event"""
        handlers = self._event_handlers[event.event_type]
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Handler {handler.__name__} failed for event {event.event_type}: {e}")
    
    def get_event_history(self, event_type: Optional[EventType] = None, 
                         limit: int = 100) -> List[NetworkEvent]:
        """Get recent event history"""
        with self._lock:
            events = self._event_history
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            return events[-limit:] if limit > 0 else events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        with self._lock:
            event_counts = defaultdict(int)
            for event in self._event_history:
                event_counts[event.event_type.value] += 1
            
            return {
                'total_events': len(self._event_history),
                'event_counts': dict(event_counts),
                'total_observers': sum(len(obs_list) for obs_list in self._observers.values()),
                'wildcard_observers': len(self._wildcard_observers),
                'total_handlers': sum(len(handlers) for handlers in self._event_handlers.values())
            }


class NetworkEventObserver(IObserver):
    """
    Base network event observer
    Template Method pattern for event handling
    """
    
    def __init__(self, name: str):
        self.name = name
        self.handled_events = 0
        
    def update(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle network event (Template Method)"""
        try:
            self.handled_events += 1
            self.pre_handle(event_type, event_data)
            self.handle_event(event_type, event_data)
            self.post_handle(event_type, event_data)
        except Exception as e:
            logger.error(f"Observer {self.name} failed to handle {event_type}: {e}")
            self.handle_error(event_type, event_data, e)
    
    def pre_handle(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Pre-processing hook"""
        pass
    
    def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Main event handling logic (to be overridden)"""
        pass
    
    def post_handle(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Post-processing hook"""
        pass
    
    def handle_error(self, event_type: str, event_data: Dict[str, Any], error: Exception) -> None:
        """Error handling hook"""
        pass


class MetricsObserver(NetworkEventObserver):
    """Observer for collecting network metrics"""
    
    def __init__(self):
        super().__init__("MetricsObserver")
        self.metrics = defaultdict(int)
    
    def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Collect metrics from events"""
        self.metrics[f"{event_type}_count"] += 1
        
        if event_type == EventType.PACKET_SENT.value:
            self.metrics["total_packets_sent"] += 1
        elif event_type == EventType.PACKET_RECEIVED.value:
            self.metrics["total_packets_received"] += 1
        elif event_type == EventType.NODE_FAILED.value:
            self.metrics["total_node_failures"] += 1
        elif event_type == EventType.HEALING_COMPLETED.value:
            self.metrics["total_healing_actions"] += 1
    
    def get_metrics(self) -> Dict[str, int]:
        """Get collected metrics"""
        return dict(self.metrics)


class LoggingObserver(NetworkEventObserver):
    """Observer for logging network events"""
    
    def __init__(self, log_level: int = logging.INFO):
        super().__init__("LoggingObserver")
        self.logger = logging.getLogger(f"{__name__}.LoggingObserver")
        self.log_level = log_level
    
    def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log network events"""
        message = f"Network Event: {event_type}"
        if 'source_node' in event_data:
            message += f" from Node {event_data['source_node']}"
        
        self.logger.log(self.log_level, message, extra={'event_data': event_data})


# Global event bus instance (Singleton pattern)
_global_event_bus: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get global event bus instance (Thread-safe Singleton)"""
    global _global_event_bus
    
    if _global_event_bus is None:
        with _event_bus_lock:
            if _global_event_bus is None:
                _global_event_bus = EventBus()
    
    return _global_event_bus


def publish_event(event_type: EventType, source_node: int, **kwargs) -> None:
    """Convenience function for publishing events"""
    event_data = {'source_node': source_node, **kwargs}
    get_event_bus().notify(event_type.value, event_data)