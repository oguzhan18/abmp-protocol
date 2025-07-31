"""
ABMP Packet Definition and Processing
Clean, immutable packet design with validation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import time
import hashlib
import json
import struct

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.interfaces import Priority
from common.exceptions import InvalidPacketError, PacketTooLargeError


class PacketType(Enum):
    """ABMP packet types"""
    DATA = 0x01
    ACK = 0x02
    DISCOVER = 0x03
    HEAL = 0x04
    PHEROMONE_UPDATE = 0x05
    TOPOLOGY_CHANGE = 0x06
    SECURITY_ALERT = 0x07
    HEARTBEAT = 0x08


@dataclass(frozen=True)
class PheromoneData:
    """Immutable pheromone data for bio-routing"""
    intensity: float
    quality_score: float
    age: float
    pheromone_type: str
    source_node: int
    
    def __post_init__(self):
        # Validation
        if not (0.0 <= self.intensity <= 1.0):
            raise InvalidPacketError(f"Pheromone intensity must be 0-1, got {self.intensity}")
        if not (0.0 <= self.quality_score <= 1.0):
            raise InvalidPacketError(f"Quality score must be 0-1, got {self.quality_score}")


@dataclass(frozen=True)
class PacketHeader:
    """Immutable ABMP packet header"""
    version: int = 1
    packet_type: PacketType = PacketType.DATA
    priority: Priority = Priority.NORMAL
    ttl: int = 64
    source_node: int = 0
    destination_node: int = 0
    sequence_number: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        # Validation
        if self.ttl < 0 or self.ttl > 255:
            raise InvalidPacketError(f"TTL must be 0-255, got {self.ttl}")
        if self.source_node < 0:
            raise InvalidPacketError(f"Source node must be >= 0, got {self.source_node}")
        if self.destination_node < 0:
            raise InvalidPacketError(f"Destination node must be >= 0, got {self.destination_node}")


@dataclass(frozen=True)
class ABMPPacket:
    """Immutable ABMP protocol packet"""
    header: PacketHeader
    payload: bytes = b''
    pheromone_data: Optional[PheromoneData] = None
    path_history: List[int] = field(default_factory=list)
    checksum: Optional[str] = field(default=None)
    
    # Class constants
    MAX_PAYLOAD_SIZE = 65536  # 64KB
    HEADER_SIZE = 64  # bytes
    
    def __post_init__(self):
        # Validate payload size
        if len(self.payload) > self.MAX_PAYLOAD_SIZE:
            raise PacketTooLargeError(len(self.payload), self.MAX_PAYLOAD_SIZE)
        
        # Calculate checksum if not provided
        if self.checksum is None:
            object.__setattr__(self, 'checksum', self._calculate_checksum())
    
    def _calculate_checksum(self) -> str:
        """Calculate packet checksum"""
        checksum_data = (
            str(self.header).encode() +
            self.payload +
            str(self.pheromone_data).encode() +
            json.dumps(self.path_history).encode()
        )
        return hashlib.sha256(checksum_data).hexdigest()[:16]
    
    def verify_checksum(self) -> bool:
        """Verify packet integrity"""
        calculated = self._calculate_checksum()
        return calculated == self.checksum
    
    def decrement_ttl(self) -> 'ABMPPacket':
        """Create new packet with decremented TTL"""
        new_header = PacketHeader(
            version=self.header.version,
            packet_type=self.header.packet_type,
            priority=self.header.priority,
            ttl=max(0, self.header.ttl - 1),
            source_node=self.header.source_node,
            destination_node=self.header.destination_node,
            sequence_number=self.header.sequence_number,
            timestamp=self.header.timestamp
        )
        
        return ABMPPacket(
            header=new_header,
            payload=self.payload,
            pheromone_data=self.pheromone_data,
            path_history=self.path_history
        )
    
    def add_to_path(self, node_id: int) -> 'ABMPPacket':
        """Create new packet with node added to path history"""
        new_path = self.path_history + [node_id]
        
        return ABMPPacket(
            header=self.header,
            payload=self.payload,
            pheromone_data=self.pheromone_data,
            path_history=new_path
        )
    
    def to_bytes(self) -> bytes:
        """Serialize packet to bytes"""
        # Header serialization
        header_data = struct.pack(
            '!BBBBQQI',
            (self.header.version << 4) | self.header.packet_type.value,
            self.header.priority.value,
            self.header.ttl,
            0,  # Reserved
            self.header.source_node,
            self.header.destination_node,
            self.header.sequence_number
        )
        
        # Pheromone data serialization
        pheromone_bytes = b'\x00' * 32
        if self.pheromone_data:
            pheromone_dict = {
                'intensity': self.pheromone_data.intensity,
                'quality_score': self.pheromone_data.quality_score,
                'age': self.pheromone_data.age,
                'type': self.pheromone_data.pheromone_type,
                'source': self.pheromone_data.source_node
            }
            pheromone_json = json.dumps(pheromone_dict).encode()[:32]
            pheromone_bytes = pheromone_json.ljust(32, b'\x00')
        
        # Path history serialization
        path_json = json.dumps(self.path_history).encode()
        path_length = struct.pack('!H', len(path_json))
        
        # Checksum
        checksum_bytes = self.checksum.encode()[:16].ljust(16, b'\x00')
        
        return (header_data + pheromone_bytes + path_length + 
                path_json + self.payload + checksum_bytes)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'ABMPPacket':
        """Deserialize packet from bytes"""
        if len(data) < cls.HEADER_SIZE:
            raise InvalidPacketError(f"Packet too small: {len(data)} < {cls.HEADER_SIZE}")
        
        # Parse header
        header_data = struct.unpack('!BBBBQQI', data[:24])
        version_type = header_data[0]
        version = (version_type >> 4) & 0xF
        packet_type = PacketType(version_type & 0xF)
        
        header = PacketHeader(
            version=version,
            packet_type=packet_type,
            priority=Priority(header_data[1]),
            ttl=header_data[2],
            source_node=header_data[4],
            destination_node=header_data[5],
            sequence_number=header_data[6]
        )
        
        # Parse pheromone data
        pheromone_data = None
        pheromone_bytes = data[24:56]
        if pheromone_bytes != b'\x00' * 32:
            try:
                pheromone_json = pheromone_bytes.rstrip(b'\x00').decode()
                pheromone_dict = json.loads(pheromone_json)
                pheromone_data = PheromoneData(
                    intensity=pheromone_dict['intensity'],
                    quality_score=pheromone_dict['quality_score'],
                    age=pheromone_dict['age'],
                    pheromone_type=pheromone_dict['type'],
                    source_node=pheromone_dict['source']
                )
            except (json.JSONDecodeError, KeyError):
                pass  # Invalid pheromone data, ignore
        
        # Parse path history
        offset = 56
        path_length = struct.unpack('!H', data[offset:offset+2])[0]
        offset += 2
        
        path_history = []
        if path_length > 0:
            try:
                path_json = data[offset:offset+path_length].decode()
                path_history = json.loads(path_json)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Invalid path data, ignore
        
        offset += path_length
        
        # Parse payload and checksum
        if len(data) >= offset + 16:
            payload = data[offset:-16]
            checksum = data[-16:].rstrip(b'\x00').decode()
        else:
            payload = data[offset:]
            checksum = ""
        
        packet = ABMPPacket(
            header=header,
            payload=payload,
            pheromone_data=pheromone_data,
            path_history=path_history,
            checksum=checksum if checksum else None
        )
        
        # Verify checksum if present
        if packet.checksum and not packet.verify_checksum():
            raise InvalidPacketError("Checksum verification failed")
        
        return packet
    
    @property
    def size(self) -> int:
        """Get packet size in bytes"""
        return len(self.to_bytes())
    
    @property
    def is_expired(self) -> bool:
        """Check if packet TTL has expired"""
        return self.header.ttl <= 0
    
    @property
    def hop_count(self) -> int:
        """Get number of hops traversed"""
        return len(self.path_history)


class PacketBuilder:
    """Builder pattern for creating ABMP packets"""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'PacketBuilder':
        """Reset builder to initial state"""
        self._header = PacketHeader()
        self._payload = b''
        self._pheromone_data = None
        self._path_history = []
        return self
    
    def with_type(self, packet_type: PacketType) -> 'PacketBuilder':
        """Set packet type"""
        self._header = PacketHeader(
            version=self._header.version,
            packet_type=packet_type,
            priority=self._header.priority,
            ttl=self._header.ttl,
            source_node=self._header.source_node,
            destination_node=self._header.destination_node,
            sequence_number=self._header.sequence_number,
            timestamp=self._header.timestamp
        )
        return self
    
    def with_priority(self, priority: Priority) -> 'PacketBuilder':
        """Set packet priority"""
        self._header = PacketHeader(
            version=self._header.version,
            packet_type=self._header.packet_type,
            priority=priority,
            ttl=self._header.ttl,
            source_node=self._header.source_node,
            destination_node=self._header.destination_node,
            sequence_number=self._header.sequence_number,
            timestamp=self._header.timestamp
        )
        return self
    
    def with_source(self, source_node: int) -> 'PacketBuilder':
        """Set source node"""
        self._header = PacketHeader(
            version=self._header.version,
            packet_type=self._header.packet_type,
            priority=self._header.priority,
            ttl=self._header.ttl,
            source_node=source_node,
            destination_node=self._header.destination_node,
            sequence_number=self._header.sequence_number,
            timestamp=self._header.timestamp
        )
        return self
    
    def with_destination(self, destination_node: int) -> 'PacketBuilder':
        """Set destination node"""
        self._header = PacketHeader(
            version=self._header.version,
            packet_type=self._header.packet_type,
            priority=self._header.priority,
            ttl=self._header.ttl,
            source_node=self._header.source_node,
            destination_node=destination_node,
            sequence_number=self._header.sequence_number,
            timestamp=self._header.timestamp
        )
        return self
    
    def with_sequence(self, sequence_number: int) -> 'PacketBuilder':
        """Set sequence number"""
        self._header = PacketHeader(
            version=self._header.version,
            packet_type=self._header.packet_type,
            priority=self._header.priority,
            ttl=self._header.ttl,
            source_node=self._header.source_node,
            destination_node=self._header.destination_node,
            sequence_number=sequence_number,
            timestamp=self._header.timestamp
        )
        return self
    
    def with_payload(self, payload: bytes) -> 'PacketBuilder':
        """Set payload data"""
        self._payload = payload
        return self
    
    def with_pheromone(self, pheromone_data: PheromoneData) -> 'PacketBuilder':
        """Set pheromone data"""
        self._pheromone_data = pheromone_data
        return self
    
    def build(self) -> ABMPPacket:
        """Build the packet"""
        packet = ABMPPacket(
            header=self._header,
            payload=self._payload,
            pheromone_data=self._pheromone_data,
            path_history=self._path_history.copy()
        )
        return packet