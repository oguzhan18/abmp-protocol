"""
ABMP Protocol Engine
Staff Engineer Level: Clean Architecture with Dependency Injection
"""

import time
import threading
import socket
import logging
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.interfaces import (
    INetworkNode, IRoutingEngine, IHealingEngine, ICryptoEngine,
    NodeState, Priority, NodeMetrics, IEventPublisher
)
from common.exceptions import (
    ABMPError, NetworkError, InvalidPacketError, 
    StateError, TimeoutError, handle_abmp_exception
)
from common.events import EventType, get_event_bus, publish_event
from core.packet import ABMPPacket, PacketBuilder, PacketType


logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """ABMP Engine configuration"""
    node_id: int
    listen_port: int = 8888
    max_connections: int = 1000
    packet_timeout: float = 30.0
    heartbeat_interval: float = 30.0
    max_packet_size: int = 65536
    enable_encryption: bool = True
    enable_healing: bool = True
    log_level: int = logging.INFO


@dataclass
class NetworkStatistics:
    """Network performance statistics"""
    packets_sent: int = 0
    packets_received: int = 0
    packets_forwarded: int = 0
    packets_dropped: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connection_count: int = 0
    uptime: float = 0.0
    last_updated: float = field(default_factory=time.time)


class ABMPEngine(INetworkNode):
    """
    Main ABMP Protocol Engine
    
    Implements:
    - Dependency Injection (Constructor Injection)
    - Single Responsibility Principle
    - Open/Closed Principle via Strategy Pattern
    - Interface Segregation via specific interfaces
    """
    
    def __init__(
        self,
        config: EngineConfig,
        routing_engine: IRoutingEngine,
        healing_engine: Optional[IHealingEngine] = None,
        crypto_engine: Optional[ICryptoEngine] = None,
        event_bus: Optional[IEventPublisher] = None
    ):
        # Dependency injection
        self._config = config
        self._routing_engine = routing_engine
        self._healing_engine = healing_engine
        self._crypto_engine = crypto_engine
        self._event_bus = event_bus or get_event_bus()
        
        # State management
        self._state = NodeState.INITIALIZING
        self._start_time = time.time()
        self._sequence_counter = 0
        self._sequence_lock = threading.Lock()
        
        # Network components
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._threads: List[threading.Thread] = []
        
        # Statistics and monitoring
        self._statistics = NetworkStatistics()
        self._stats_lock = threading.RLock()
        
        # Packet management
        self._pending_packets: Dict[str, ABMPPacket] = {}
        self._packet_callbacks: Dict[str, Callable[[ABMPPacket], None]] = {}
        self._packet_lock = threading.RLock()
        
        # Connection management
        self._active_connections: Set[tuple] = set()
        self._connection_lock = threading.RLock()
        
        logger.info(f"ABMP Engine initialized for Node {self.node_id}")
    
    @property
    def node_id(self) -> int:
        """Get node identifier"""
        return self._config.node_id
    
    @property
    def state(self) -> NodeState:
        """Get current node state"""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if engine is running"""
        return self._running and self._state == NodeState.ACTIVE
    
    @handle_abmp_exception
    def start(self) -> None:
        """Start the ABMP engine"""
        if self._state != NodeState.INITIALIZING:
            raise StateError(self._state.value, "start")
        
        logger.info(f"Starting ABMP Engine for Node {self.node_id}")
        
        try:
            # Initialize network socket
            self._initialize_socket()
            
            # Start background threads
            self._start_background_threads()
            
            # Update state
            self._state = NodeState.ACTIVE
            self._running = True
            
            # Publish startup event
            publish_event(EventType.NODE_JOINED, self.node_id)
            
            logger.info(f"ABMP Engine started successfully for Node {self.node_id}")
            
        except Exception as e:
            self._state = NodeState.FAILED
            logger.error(f"Failed to start ABMP Engine: {e}")
            raise NetworkError(f"Engine startup failed: {e}") from e
    
    @handle_abmp_exception
    def stop(self) -> None:
        """Stop the ABMP engine"""
        if self._state not in [NodeState.ACTIVE, NodeState.IDLE]:
            return
        
        logger.info(f"Stopping ABMP Engine for Node {self.node_id}")
        
        # Update state
        self._state = NodeState.SHUTDOWN
        self._running = False
        
        # Stop background threads
        self._stop_background_threads()
        
        # Close socket
        if self._socket:
            self._socket.close()
            self._socket = None
        
        # Publish shutdown event
        publish_event(EventType.NODE_LEFT, self.node_id)
        
        logger.info(f"ABMP Engine stopped for Node {self.node_id}")
    
    @handle_abmp_exception
    def send_message(self, destination: int, data: bytes, priority: Priority = Priority.NORMAL) -> bool:
        """Send message to destination node"""
        if not self.is_running:
            raise StateError(self._state.value, "send_message")
        
        if len(data) > self._config.max_packet_size:
            raise InvalidPacketError(f"Data too large: {len(data)} > {self._config.max_packet_size}")
        
        # Build packet
        packet = (PacketBuilder()
                 .with_type(PacketType.DATA)
                 .with_source(self.node_id)
                 .with_destination(destination)
                 .with_priority(priority)
                 .with_sequence(self._next_sequence())
                 .with_payload(data)
                 .build())
        
        # Route and send packet
        return self._route_packet(packet)
    
    @handle_abmp_exception
    def get_metrics(self) -> NodeMetrics:
        """Get current node metrics"""
        with self._stats_lock:
            return NodeMetrics(
                node_id=self.node_id,
                cpu_usage=self._get_cpu_usage(),
                memory_usage=self._get_memory_usage(),
                network_latency=self._get_network_latency(),
                packet_loss_rate=self._calculate_packet_loss_rate(),
                uptime=time.time() - self._start_time
            )
    
    def get_statistics(self) -> Dict[str, any]:
        """Get detailed engine statistics"""
        with self._stats_lock:
            stats = {
                'node_id': self.node_id,
                'state': self._state.value,
                'uptime': time.time() - self._start_time,
                'packets_sent': self._statistics.packets_sent,
                'packets_received': self._statistics.packets_received,
                'packets_forwarded': self._statistics.packets_forwarded,
                'packets_dropped': self._statistics.packets_dropped,
                'bytes_sent': self._statistics.bytes_sent,
                'bytes_received': self._statistics.bytes_received,
                'active_connections': len(self._active_connections),
                'pending_packets': len(self._pending_packets)
            }
            
            # Add routing engine stats if available
            if hasattr(self._routing_engine, 'get_statistics'):
                stats['routing'] = self._routing_engine.get_statistics()
            
            # Add healing engine stats if available
            if self._healing_engine and hasattr(self._healing_engine, 'get_statistics'):
                stats['healing'] = self._healing_engine.get_statistics()
            
            return stats
    
    def _initialize_socket(self) -> None:
        """Initialize network socket"""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(('0.0.0.0', self._config.listen_port))
        self._socket.settimeout(1.0)  # Non-blocking with timeout
        
        logger.debug(f"Socket initialized on port {self._config.listen_port}")
    
    def _start_background_threads(self) -> None:
        """Start background worker threads"""
        threads = [
            ('packet_listener', self._packet_listener_loop),
            ('heartbeat_sender', self._heartbeat_loop),
            ('statistics_updater', self._statistics_loop),
            ('packet_cleanup', self._packet_cleanup_loop)
        ]
        
        for name, target in threads:
            thread = threading.Thread(target=target, name=f"ABMP-{name}-{self.node_id}", daemon=True)
            thread.start()
            self._threads.append(thread)
        
        logger.debug(f"Started {len(threads)} background threads")
    
    def _stop_background_threads(self) -> None:
        """Stop background threads"""
        # Threads will stop when self._running becomes False
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self._threads.clear()
        logger.debug("Background threads stopped")
    
    @contextmanager
    def _update_statistics(self):
        """Context manager for thread-safe statistics updates"""
        with self._stats_lock:
            self._statistics.last_updated = time.time()
            yield self._statistics
    
    def _next_sequence(self) -> int:
        """Get next sequence number (thread-safe)"""
        with self._sequence_lock:
            self._sequence_counter += 1
            return self._sequence_counter
    
    def _route_packet(self, packet: ABMPPacket) -> bool:
        """Route packet using routing engine"""
        try:
            # Check if destination is self
            if packet.header.destination_node == self.node_id:
                return self._handle_local_packet(packet)
            
            # Check TTL
            if packet.is_expired:
                logger.warning(f"Packet TTL expired: {packet.header.sequence_number}")
                with self._update_statistics() as stats:
                    stats.packets_dropped += 1
                return False
            
            # Get routing decision
            current_metrics = self.get_metrics()
            routing_decision = self._routing_engine.find_route(
                packet.header.destination_node, current_metrics
            )
            
            if not routing_decision.next_hop:
                logger.warning(f"No route found to {packet.header.destination_node}")
                with self._update_statistics() as stats:
                    stats.packets_dropped += 1
                return False
            
            # Forward packet
            return self._forward_packet(packet, routing_decision.next_hop)
            
        except Exception as e:
            logger.error(f"Packet routing failed: {e}")
            with self._update_statistics() as stats:
                stats.packets_dropped += 1
            return False
    
    def _forward_packet(self, packet: ABMPPacket, next_hop: int) -> bool:
        """Forward packet to next hop"""
        try:
            # Decrement TTL and add to path
            forwarded_packet = packet.decrement_ttl().add_to_path(self.node_id)
            
            # Serialize and send
            packet_bytes = forwarded_packet.to_bytes()
            
            # TODO: Get next hop address from topology
            # For now, use port offset (demo purposes)
            target_port = 8880 + next_hop
            target_address = ('127.0.0.1', target_port)
            
            self._socket.sendto(packet_bytes, target_address)
            
            with self._update_statistics() as stats:
                stats.packets_forwarded += 1
                stats.bytes_sent += len(packet_bytes)
            
            # Publish routing event
            publish_event(EventType.PACKET_SENT, self.node_id,
                         destination=packet.header.destination_node,
                         next_hop=next_hop)
            
            return True
            
        except Exception as e:
            logger.error(f"Packet forwarding failed: {e}")
            return False
    
    def _handle_local_packet(self, packet: ABMPPacket) -> bool:
        """Handle packet destined for this node"""
        try:
            logger.debug(f"Handling local packet type {packet.header.packet_type}")
            
            with self._update_statistics() as stats:
                stats.packets_received += 1
                stats.bytes_received += packet.size
            
            # Handle different packet types
            if packet.header.packet_type == PacketType.DATA:
                self._handle_data_packet(packet)
            elif packet.header.packet_type == PacketType.HEARTBEAT:
                self._handle_heartbeat_packet(packet)
            elif packet.header.packet_type == PacketType.DISCOVER:
                self._handle_discovery_packet(packet)
            elif packet.header.packet_type == PacketType.HEAL:
                self._handle_healing_packet(packet)
            
            # Publish receive event
            publish_event(EventType.PACKET_RECEIVED, self.node_id,
                         source=packet.header.source_node,
                         packet_type=packet.header.packet_type.name)
            
            return True
            
        except Exception as e:
            logger.error(f"Local packet handling failed: {e}")
            return False
    
    def _handle_data_packet(self, packet: ABMPPacket) -> None:
        """Handle data packet"""
        # Decrypt if encryption is enabled
        payload = packet.payload
        if self._crypto_engine and self._config.enable_encryption:
            try:
                payload = self._crypto_engine.decrypt(payload, packet.header.source_node)
            except Exception as e:
                logger.error(f"Packet decryption failed: {e}")
                return
        
        # Process payload (application-specific logic)
        logger.info(f"Received data from Node {packet.header.source_node}: {len(payload)} bytes")
    
    def _handle_heartbeat_packet(self, packet: ABMPPacket) -> None:
        """Handle heartbeat packet"""
        # Update routing table with node liveness
        metrics = NodeMetrics(
            node_id=packet.header.source_node,
            cpu_usage=0.0,  # Will be updated with real metrics
            memory_usage=0.0,
            network_latency=0.0,
            packet_loss_rate=0.0,
            uptime=0.0
        )
        self._routing_engine.update_routing_table(packet.header.source_node, metrics)
    
    def _handle_discovery_packet(self, packet: ABMPPacket) -> None:
        """Handle node discovery packet"""
        logger.info(f"Node {packet.header.source_node} discovered")
        # TODO: Implement discovery protocol
    
    def _handle_healing_packet(self, packet: ABMPPacket) -> None:
        """Handle healing packet"""
        if self._healing_engine:
            # TODO: Process healing information
            pass
    
    def _packet_listener_loop(self) -> None:
        """Background packet listener thread"""
        logger.debug("Packet listener started")
        
        while self._running:
            try:
                data, addr = self._socket.recvfrom(self._config.max_packet_size)
                
                # Parse packet
                packet = ABMPPacket.from_bytes(data)
                
                # Track connection
                with self._connection_lock:
                    self._active_connections.add(addr)
                
                # Route packet in separate thread to avoid blocking
                threading.Thread(
                    target=self._route_packet,
                    args=(packet,),
                    daemon=True
                ).start()
                
            except socket.timeout:
                continue  # Normal timeout, continue listening
            except Exception as e:
                if self._running:  # Only log if we're still supposed to be running
                    logger.error(f"Packet listener error: {e}")
                
        logger.debug("Packet listener stopped")
    
    def _heartbeat_loop(self) -> None:
        """Background heartbeat sender thread"""
        logger.debug("Heartbeat sender started")
        
        while self._running:
            try:
                # Send heartbeat to all known nodes
                # TODO: Get known nodes from topology
                
                time.sleep(self._config.heartbeat_interval)
                
            except Exception as e:
                if self._running:
                    logger.error(f"Heartbeat sender error: {e}")
        
        logger.debug("Heartbeat sender stopped")
    
    def _statistics_loop(self) -> None:
        """Background statistics updater thread"""
        logger.debug("Statistics updater started")
        
        while self._running:
            try:
                # Update metrics if healing engine is available
                if self._healing_engine:
                    metrics = self.get_metrics()
                    # Report metrics to healing engine for anomaly detection
                    # This is where the healing engine would check for anomalies
                
                time.sleep(10.0)  # Update every 10 seconds
                
            except Exception as e:
                if self._running:
                    logger.error(f"Statistics updater error: {e}")
        
        logger.debug("Statistics updater stopped")
    
    def _packet_cleanup_loop(self) -> None:
        """Background packet cleanup thread"""
        logger.debug("Packet cleanup started")
        
        while self._running:
            try:
                current_time = time.time()
                expired_packets = []
                
                with self._packet_lock:
                    for packet_id, packet in self._pending_packets.items():
                        if current_time - packet.header.timestamp > self._config.packet_timeout:
                            expired_packets.append(packet_id)
                    
                    for packet_id in expired_packets:
                        del self._pending_packets[packet_id]
                        self._packet_callbacks.pop(packet_id, None)
                
                if expired_packets:
                    logger.debug(f"Cleaned up {len(expired_packets)} expired packets")
                
                time.sleep(30.0)  # Cleanup every 30 seconds
                
            except Exception as e:
                if self._running:
                    logger.error(f"Packet cleanup error: {e}")
        
        logger.debug("Packet cleanup stopped")
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def _get_network_latency(self) -> float:
        """Get current network latency"""
        # TODO: Implement actual latency measurement
        return 0.0
    
    def _calculate_packet_loss_rate(self) -> float:
        """Calculate packet loss rate"""
        if self._statistics.packets_sent == 0:
            return 0.0
        
        return self._statistics.packets_dropped / self._statistics.packets_sent